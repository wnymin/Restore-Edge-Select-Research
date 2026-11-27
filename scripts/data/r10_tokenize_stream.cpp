// Length-prefixed binary pipe using the frozen llama.cpp vocabulary, without
// loading model tensors, running an inference server, or opening a socket.
#include "llama.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
constexpr uint32_t max_frame = 1 << 20;
void error(const char *message) { std::fprintf(stderr, "r10 tokenization: %s\n", message); }
}

int main(int argc, char **argv) {
    if (argc != 3) { error("usage: r10_tokenize_stream MODEL_GGUF CPU_BACKEND_LIBRARY"); return 1; }
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1) {
        error("could not use binary pipes"); return 1;
    }
#endif
    if (!ggml_backend_load(argv[2])) { error("could not register CPU backend"); return 1; }
    llama_backend_init();
    struct Cleanup { ~Cleanup() { llama_backend_free(); } } cleanup;
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], params), llama_model_free);
    if (!model) { error("could not load the vocabulary"); return 1; }
    const auto *vocab = llama_model_get_vocab(model.get());
    if (!vocab || llama_vocab_n_tokens(vocab) < 1 || sizeof(llama_token) != sizeof(uint32_t)) {
        error("unexpected vocabulary or wire format"); return 1;
    }
    for (;;) {
        uint32_t n = 0;
        const size_t got = std::fread(&n, 1, sizeof(n), stdin);
        if (!got && std::feof(stdin)) return 0;
        if (got != sizeof(n) || n == 0 || n > max_frame) { error("invalid frame length"); return 1; }
        std::string prompt(n, '\0');
        if (std::fread(prompt.data(), 1, n, stdin) != n) { error("short prompt frame"); return 1; }
        std::vector<llama_token> ids(n);
        const auto nt = llama_tokenize(vocab, prompt.data(), int32_t(n), ids.data(), int32_t(ids.size()), false, true);
        if (nt <= 0 || size_t(nt) > ids.size()) { error("tokenization failed"); return 1; }
        const uint32_t count = uint32_t(nt);
        if (std::fwrite(&count, 1, sizeof(count), stdout) != sizeof(count) ||
            std::fwrite(ids.data(), sizeof(llama_token), count, stdout) != count || std::fflush(stdout) != 0) {
            error("incomplete token frame"); return 1;
        }
    }
}
