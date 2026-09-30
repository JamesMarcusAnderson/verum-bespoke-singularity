#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tokenizer_t tokenizer_t;

tokenizer_t *tokenizer_load(const char *path);
size_t tokenizer_encode(tokenizer_t *tok, const char *text, uint32_t *out_tokens, size_t max_tokens);
char *tokenizer_decode(tokenizer_t *tok, const uint32_t *tokens, size_t n_tokens);
void tokenizer_free(tokenizer_t *tok);

#ifdef __cplusplus
}
#endif
