#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

typedef struct { const char *p; const char *end; } json_reader_t;
static void skip_ws(json_reader_t *r) { while (r->p < r->end && isspace((unsigned char)*r->p)) r->p++; }
static int expect_char(json_reader_t *r, char c) { skip_ws(r); if (r->p < r->end && *r->p == c) { r->p++; return 1; } return 0; }
static char *read_string(json_reader_t *r) {
    if (!expect_char(r, '"')) return NULL;
    const char *start = r->p;
    while (r->p < r->end && *r->p != '"') { if (*r->p == '\\') r->p++; r->p++; }
    if (r->p >= r->end) return NULL;
    size_t len = r->p - start;
    char *s = (char *)malloc(len + 1);
    memcpy(s, start, len); s[len] = 0; r->p++; return s;
}
static int64_t read_int(json_reader_t *r) { skip_ws(r); char *end; int64_t v = strtoll(r->p, &end, 10); r->p = end; return v; }
static void skip_value(json_reader_t *r);
static void skip_array(json_reader_t *r) {
    if (!expect_char(r, '[')) return;
    while (r->p < r->end && *r->p != ']') { skip_value(r); skip_ws(r); if (*r->p == ',') r->p++; }
    expect_char(r, ']');
}
static void skip_object(json_reader_t *r) {
    if (!expect_char(r, '{')) return;
    while (r->p < r->end && *r->p != '}') { char *k = read_string(r); free(k); expect_char(r, ':'); skip_value(r); skip_ws(r); if (*r->p == ',') r->p++; }
    expect_char(r, '}');
}
static void skip_value(json_reader_t *r) {
    skip_ws(r); if (r->p >= r->end) return;
    switch (*r->p) { case '"': { char *s = read_string(r); free(s); break; } case '[': skip_array(r); break; case '{': skip_object(r); break; default: while (r->p < r->end && !isspace((unsigned char)*r->p) && !strchr(",]}", *r->p)) r->p++; }
}

#define MAX_VOCAB 200000
#define MAX_MERGES 200000
typedef struct { char *text; int len; } vocab_entry_t;
typedef struct { uint32_t a, b, merged; } merge_t;

struct tokenizer_t {
    vocab_entry_t vocab[MAX_VOCAB]; int vocab_size;
    merge_t merges[MAX_MERGES]; int merges_size;
    int byte_to_token[256];
    int eos_token_id, bos_token_id;
};

static int parse_vocab(json_reader_t *r, tokenizer_t *tok) {
    if (!expect_char(r, '{')) return 0;
    while (r->p < r->end && *r->p != '}') {
        char *text = read_string(r); if (!text) break;
        expect_char(r, ':'); int64_t id = read_int(r);
        if (id >= 0 && id < MAX_VOCAB) { tok->vocab[id].text = strdup(text); tok->vocab[id].len = (int)strlen(text); if (id >= tok->vocab_size) tok->vocab_size = (int)id + 1; }
        free(text); skip_ws(r); if (*r->p == ',') r->p++;
    }
    expect_char(r, '}'); return 1;
}

static int parse_merges(json_reader_t *r, tokenizer_t *tok) {
    if (!expect_char(r, '[')) return 0;
    while (r->p < r->end && *r->p != ']') {
        char *m = read_string(r); if (!m) break;
        char *sp = strchr(m, ' '); if (sp) { *sp = 0;
            int id_a = -1, id_b = -1, merged_id = -1;
            for (int i=0;i<tok->vocab_size;i++) if (tok->vocab[i].text && strcmp(tok->vocab[i].text,m)==0) { id_a=i; break; }
            for (int i=0;i<tok->vocab_size;i++) if (tok->vocab[i].text && strcmp(tok->vocab[i].text,sp+1)==0) { id_b=i; break; }
            char *mt = strdup(m); strcat(mt, sp+1);
            for (int i=0;i<tok->vocab_size;i++) if (tok->vocab[i].text && strcmp(tok->vocab[i].text,mt)==0) { merged_id=i; break; }
            free(mt);
            if (id_a>=0 && id_b>=0 && merged_id>=0 && tok->merges_size<MAX_MERGES) {
                tok->merges[tok->merges_size] = (merge_t){(uint32_t)id_a, (uint32_t)id_b, (uint32_t)merged_id}; tok->merges_size++;
            }
        }
        free(m); skip_ws(r); if (*r->p == ',') r->p++;
    }
    expect_char(r, ']'); return 1;
}

tokenizer_t *tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f,0,SEEK_END); size_t len = ftell(f); fseek(f,0,SEEK_SET);
    char *buf = (char *)malloc(len+1); fread(buf,1,len,f); buf[len]=0; fclose(f);
    tokenizer_t *tok = (tokenizer_t *)calloc(1,sizeof(tokenizer_t));
    json_reader_t r = { buf, buf+len };
    expect_char(&r,'{');
    while (r.p < r.end && *r.p != '}') {
        char *key = read_string(&r); if (!key) break; expect_char(&r,':');
        if (strcmp(key,"model")==0) {
            expect_char(&r,'{');
            while (r.p < r.end && *r.p != '}') {
                char *ik = read_string(&r); if (!ik) break; expect_char(&r,':');
                if (strcmp(ik,"vocab")==0) parse_vocab(&r,tok);
                else if (strcmp(ik,"merges")==0) parse_merges(&r,tok);
                else skip_value(&r);
                free(ik); skip_ws(&r); if (*r.p==',') r.p++;
            }
            expect_char(&r,'}');
        } else skip_value(&r);
        free(key); skip_ws(&r); if (*r.p==',') r.p++;
    }
    free(buf);
    for (int i=0;i<tok->vocab_size;i++) if (tok->vocab[i].text && strcmp(tok->vocab[i].text,"<|endoftext|>")==0) { tok->eos_token_id=i; tok->bos_token_id=i; break; }
    for (int i=0;i<256;i++) tok->byte_to_token[i]=-1;
    for (int i=0;i<tok->vocab_size;i++) if (tok->vocab[i].text && tok->vocab[i].len==1) {
        unsigned char c = (unsigned char)tok->vocab[i].text[0];
        if (c<128) tok->byte_to_token[c]=i;
    }
    return tok;
}

static void find_best_pair(const uint32_t *tokens, int n, const tokenizer_t *tok, int *best_pos, uint32_t *best_a, uint32_t *best_b, uint32_t *best_merged) {
    for (int m=0; m<tok->merges_size; m++) {
        uint32_t a = tok->merges[m].a, b = tok->merges[m].b;
        for (int i=0; i<n-1; i++) if (tokens[i]==a && tokens[i+1]==b) { *best_pos=i; *best_a=a; *best_b=b; *best_merged=tok->merges[m].merged; return; }
    }
    *best_pos=-1;
}

size_t tokenizer_encode(tokenizer_t *tok, const char *text, uint32_t *out_tokens, size_t max_tokens) {
    if (!tok||!text) return 0;
    #define MAX_INIT 4096
    uint32_t init[MAX_INIT]; int n_init=0;
    for (const char *p=text; *p && n_init<MAX_INIT; p++) {
        unsigned char byte = *p; int tid = tok->byte_to_token[byte]; if (tid<0) tid=0;
        init[n_init++] = tid;
    }
    int n = n_init; uint32_t *tokens = init;
    while (n>=2) { int bp=-1; uint32_t ba,bb,bm; find_best_pair(tokens,n,tok,&bp,&ba,&bb,&bm); if (bp<0) break; tokens[bp]=bm; for (int i=bp+1;i<n-1;i++) tokens[i]=tokens[i+1]; n--; }
    size_t out_n = (size_t)n < max_tokens ? (size_t)n : max_tokens;
    if (out_tokens) memcpy(out_tokens, tokens, out_n*sizeof(uint32_t));
    return n;
}

char *tokenizer_decode(tokenizer_t *tok, const uint32_t *tokens, size_t n_tokens) {
    size_t total=0;
    for (size_t i=0;i<n_tokens;i++) { uint32_t id=tokens[i]; if (id<tok->vocab_size && tok->vocab[id].text) total+=tok->vocab[id].len; }
    char *r=(char*)malloc(total+1), *p=r;
    for (size_t i=0;i<n_tokens;i++) { uint32_t id=tokens[i]; if (id<tok->vocab_size && tok->vocab[id].text) { memcpy(p,tok->vocab[id].text,tok->vocab[id].len); p+=tok->vocab[id].len; } }
    *p=0; return r;
}

void tokenizer_free(tokenizer_t *tok) { if (!tok) return; for (int i=0;i<tok->vocab_size;i++) free(tok->vocab[i].text); free(tok); }

