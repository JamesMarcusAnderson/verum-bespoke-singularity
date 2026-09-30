#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <ctype.h>

typedef struct { const char *p, *end; } jr_t;

static void jr_ws(jr_t *r) {
    while (r->p < r->end && (*r->p == ' ' || *r->p == '\t' || *r->p == '\n' || *r->p == '\r')) r->p++;
}

static int jr_ch(jr_t *r, char c) {
    jr_ws(r);
    if (r->p < r->end && *r->p == c) { r->p++; return 1; }
    return 0;
}

static int utf8_encode_cp(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static uint32_t jr_hex4(jr_t *r) {
    uint32_t v = 0;
    for (int i = 0; i < 4 && r->p < r->end; i++, r->p++) {
        int d = hexval(*r->p);
        if (d < 0) break;
        v = v * 16 + (uint32_t)d;
    }
    return v;
}

static char *jr_string(jr_t *r, size_t *out_len) {
    if (!jr_ch(r, '"')) return NULL;
    size_t cap = 64, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    while (r->p < r->end && *r->p != '"') {
        if (len + 8 > cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        char c = *r->p++;
        if (c == '\\' && r->p < r->end) {
            char e = *r->p++;
            switch (e) {
                case '"':  buf[len++] = '"';  break;
                case '\\': buf[len++] = '\\'; break;
                case '/':  buf[len++] = '/';  break;
                case 'b':  buf[len++] = '\b'; break;
                case 'f':  buf[len++] = '\f'; break;
                case 'n':  buf[len++] = '\n'; break;
                case 'r':  buf[len++] = '\r'; break;
                case 't':  buf[len++] = '\t'; break;
                case 'u': {
                    if (r->end - r->p < 4) { free(buf); return NULL; }
                    uint32_t cp = jr_hex4(r);
                    if (cp >= 0xD800 && cp <= 0xDBFF &&
                        r->end - r->p >= 6 && r->p[0] == '\\' && r->p[1] == 'u') {
                        r->p += 2;
                        uint32_t lo = jr_hex4(r);
                        if (lo >= 0xDC00 && lo <= 0xDFFF)
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    len += (size_t)utf8_encode_cp(cp, buf + len);
                } break;
                default: buf[len++] = e; break;
            }
        } else {
            buf[len++] = c;
        }
    }
    if (r->p >= r->end) { free(buf); return NULL; }
    r->p++;
    if (len + 1 > cap) {
        char *nb = (char *)realloc(buf, len + 1);
        if (!nb) { free(buf); return NULL; }
        buf = nb;
    }
    buf[len] = 0;
    if (out_len) *out_len = len;
    return buf;
}

static void jr_skip_value(jr_t *r);

static void jr_skip_array(jr_t *r) {
    if (!jr_ch(r, '[')) return;
    while (r->p < r->end && *r->p != ']') {
        jr_skip_value(r);
        jr_ws(r);
        if (r->p < r->end && *r->p == ',') r->p++;
    }
    jr_ch(r, ']');
}

static void jr_skip_object(jr_t *r) {
    if (!jr_ch(r, '{')) return;
    while (r->p < r->end && *r->p != '}') {
        size_t kl;
        char *k = jr_string(r, &kl);
        free(k);
        jr_ch(r, ':');
        jr_skip_value(r);
        jr_ws(r);
        if (r->p < r->end && *r->p == ',') r->p++;
    }
    jr_ch(r, '}');
}

static void jr_skip_value(jr_t *r) {
    jr_ws(r);
    if (r->p >= r->end) return;
    switch (*r->p) {
        case '"': { size_t l; char *s = jr_string(r, &l); free(s); break; }
        case '[': jr_skip_array(r); break;
        case '{': jr_skip_object(r); break;
        default:
            while (r->p < r->end && !isspace((unsigned char)*r->p) && !strchr(",]}", *r->p)) r->p++;
    }
}

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

typedef struct { char *key; size_t len; int id; } strslot_t;
typedef struct { strslot_t *slots; size_t cap; } strmap_t;

static size_t pow2_at_least(size_t n) {
    size_t c = 64;
    while (c < n) c <<= 1;
    return c;
}

static void strmap_init(strmap_t *m, size_t expected) {
    m->cap = pow2_at_least(expected * 2 + 16);
    m->slots = (strslot_t *)calloc(m->cap, sizeof(strslot_t));
    for (size_t i = 0; i < m->cap; i++) m->slots[i].id = -1;
}

static void strmap_put(strmap_t *m, const char *key, size_t len, int id) {
    uint64_t h = fnv1a(key, len);
    size_t idx = (size_t)(h & (m->cap - 1));
    while (m->slots[idx].key != NULL) {
        if (m->slots[idx].len == len && memcmp(m->slots[idx].key, key, len) == 0) return;
        idx = (idx + 1) & (m->cap - 1);
    }
    m->slots[idx].key = (char *)malloc(len + 1);
    memcpy(m->slots[idx].key, key, len);
    m->slots[idx].key[len] = 0;
    m->slots[idx].len = len;
    m->slots[idx].id = id;
}

static int strmap_get(const strmap_t *m, const char *key, size_t len) {
    uint64_t h = fnv1a(key, len);
    size_t idx = (size_t)(h & (m->cap - 1));
    while (m->slots[idx].key != NULL) {
        if (m->slots[idx].len == len && memcmp(m->slots[idx].key, key, len) == 0)
            return m->slots[idx].id;
        idx = (idx + 1) & (m->cap - 1);
    }
    return -1;
}

typedef struct { uint64_t key; int rank; uint32_t merged; } mslot_t;
typedef struct { mslot_t *slots; size_t cap; } mergemap_t;
#define MERGE_EMPTY UINT64_MAX

static void mergemap_init(mergemap_t *m, size_t expected) {
    m->cap = pow2_at_least(expected * 2 + 16);
    m->slots = (mslot_t *)calloc(m->cap, sizeof(mslot_t));
    for (size_t i = 0; i < m->cap; i++) m->slots[i].key = MERGE_EMPTY;
}

static void mergemap_put(mergemap_t *m, uint32_t a, uint32_t b, int rank, uint32_t merged) {
    uint64_t k = ((uint64_t)a << 32) | (uint64_t)b;
    size_t idx = (size_t)((k * 11400714819323198485ULL) & (m->cap - 1));
    while (m->slots[idx].key != MERGE_EMPTY) idx = (idx + 1) & (m->cap - 1);
    m->slots[idx].key = k;
    m->slots[idx].rank = rank;
    m->slots[idx].merged = merged;
}

static int mergemap_get(const mergemap_t *m, uint32_t a, uint32_t b, uint32_t *merged) {
    uint64_t k = ((uint64_t)a << 32) | (uint64_t)b;
    size_t idx = (size_t)((k * 11400714819323198485ULL) & (m->cap - 1));
    while (m->slots[idx].key != MERGE_EMPTY) {
        if (m->slots[idx].key == k) {
            if (merged) *merged = m->slots[idx].merged;
            return m->slots[idx].rank;
        }
        idx = (idx + 1) & (m->cap - 1);
    }
    return -1;
}

typedef struct { char *text; size_t len; int byte_value; } vocab_entry_t;

struct tokenizer_t {
    vocab_entry_t *vocab;
    int vocab_size, vocab_cap;
    strmap_t text2id;
    mergemap_t merges;
    int byte_to_token[256];
    int eos_token_id;
};

static void vocab_put(tokenizer_t *tok, int id, const char *text, size_t len) {
    if (id < 0 || id >= tok->vocab_cap) return;
    free(tok->vocab[id].text);
    tok->vocab[id].text = (char *)malloc(len + 1);
    memcpy(tok->vocab[id].text, text, len);
    tok->vocab[id].text[len] = 0;
    tok->vocab[id].len = len;
    tok->vocab[id].byte_value = -1;
    if (id >= tok->vocab_size) tok->vocab_size = id + 1;
    strmap_put(&tok->text2id, text, len, id);
}

static int parse_vocab_object(jr_t *r, tokenizer_t *tok) {
    if (!jr_ch(r, '{')) return 0;
    while (r->p < r->end && *r->p != '}') {
        size_t tl;
        char *text = jr_string(r, &tl);
        if (!text) break;
        jr_ch(r, ':');
        jr_ws(r);
        long long id = strtoll(r->p, (char **)&r->p, 10);
        vocab_put(tok, (int)id, text, tl);
        free(text);
        jr_ws(r);
        if (r->p < r->end && *r->p == ',') r->p++;
    }
    jr_ch(r, '}');
    return 1;
}

static int parse_merges_array(jr_t *r, tokenizer_t *tok) {
    if (!jr_ch(r, '[')) return 0;
    int rank = 0;
    while (r->p < r->end && *r->p != ']') {
        size_t ml;
        char *m = jr_string(r, &ml);
        if (!m) break;
        const char *sp = (const char *)memchr(m, ' ', ml);
        if (sp && sp > m && sp < m + ml - 1) {
            size_t la = (size_t)(sp - m), lb = ml - la - 1;
            int id_a = strmap_get(&tok->text2id, m, la);
            int id_b = strmap_get(&tok->text2id, sp + 1, lb);
            char *mt = (char *)malloc(la + lb + 1);
            memcpy(mt, m, la);
            memcpy(mt + la, sp + 1, lb);
            mt[la + lb] = 0;
            int id_m = strmap_get(&tok->text2id, mt, la + lb);
            free(mt);
            if (id_a >= 0 && id_b >= 0 && id_m >= 0)
                mergemap_put(&tok->merges, (uint32_t)id_a, (uint32_t)id_b, rank, (uint32_t)id_m);
            rank++;
        }
        free(m);
        jr_ws(r);
        if (r->p < r->end && *r->p == ',') r->p++;
    }
    jr_ch(r, ']');
    return 1;
}

tokenizer_t *tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tokenizer: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    if (flen <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)flen + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)flen, f) != (size_t)flen) { fclose(f); free(buf); return NULL; }
    buf[flen] = 0;
    fclose(f);

    tokenizer_t *tok = (tokenizer_t *)calloc(1, sizeof(tokenizer_t));
    tok->vocab_cap = 200000;
    tok->vocab = (vocab_entry_t *)calloc((size_t)tok->vocab_cap, sizeof(vocab_entry_t));
    for (int i = 0; i < tok->vocab_cap; i++) tok->vocab[i].byte_value = -1;
    strmap_init(&tok->text2id, (size_t)tok->vocab_cap);
    mergemap_init(&tok->merges, 160000);
    tok->eos_token_id = -1;

    jr_t r = { buf, buf + flen };
    jr_ch(&r, '{');
    while (r.p < r.end && *r.p != '}') {
        size_t kl;
        char *key = jr_string(&r, &kl);
        if (!key) break;
        jr_ch(&r, ':');
        if (strcmp(key, "model") == 0) {
            jr_ch(&r, '{');
            while (r.p < r.end && *r.p != '}') {
                size_t ikl;
                char *ik = jr_string(&r, &ikl);
                if (!ik) break;
                jr_ch(&r, ':');
                if (strcmp(ik, "vocab") == 0) parse_vocab_object(&r, tok);
                else if (strcmp(ik, "merges") == 0) parse_merges_array(&r, tok);
                else jr_skip_value(&r);
                free(ik);
                jr_ws(&r);
                if (r.p < r.end && *r.p == ',') r.p++;
            }
            jr_ch(&r, '}');
        } else {
            jr_skip_value(&r);
        }
        free(key);
        jr_ws(&r);
        if (r.p < r.end && *r.p == ',') r.p++;
    }
    free(buf);

    for (int i = 0; i < 256; i++) tok->byte_to_token[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t cp;
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        if (printable) cp = (uint32_t)b;
        else cp = (uint32_t)(256 + n++);
        char u8[4];
        int l = utf8_encode_cp(cp, u8);
        int id = strmap_get(&tok->text2id, u8, (size_t)l);
        tok->byte_to_token[b] = id;
        if (id >= 0 && id < tok->vocab_size) tok->vocab[id].byte_value = b;
    }

        int eos = strmap_get(&tok->text2id, "<|endoftext|>", 13);
    if (eos < 0) eos = strmap_get(&tok->text2id, "<|im_end|>", 11);
    tok->eos_token_id = eos;
    return tok;
}

static int lcp_ci(const char *a, const char *b) {
    int i = 0;
    while (a[i] && b[i] && tolower((unsigned char)a[i]) == tolower((unsigned char)b[i])) i++;
    return (b[i] == 0) ? i : -1;
}

static int contraction_len(const char *s) {
    static const char *sufs[] = { "re", "ve", "ll", "s", "t", "m", "d" };
    if (s[0] != '\'' && s[0] != '\xe2') return 0;
    char norm[16];
    size_t w = 0;
    if (s[0] == '\'') {
        norm[w++] = '\'';
        for (size_t i = 1; s[i] && w < 14; i++) norm[w++] = (char)tolower((unsigned char)s[i]);
    } else {
        if ((unsigned char)s[1] != 0x80 || (unsigned char)s[2] != 0x99) return 0;
        norm[w++] = '\'';
        for (size_t i = 3; s[i] && w < 14; i++) norm[w++] = (char)tolower((unsigned char)s[i]);
    }
    norm[w] = 0;
    for (size_t k = 0; k < sizeof(sufs) / sizeof(sufs[0]); k++) {
        int m = lcp_ci(norm, sufs[k]);
        if (m > 0) return m;
    }
    return 0;
}

static size_t utf8_len_of(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static size_t segment_len_at(const char *p) {
    unsigned char c = (unsigned char)*p;
    size_t n = 0;
    if (c == '\r' || c == '\n') {
        while (p[n] == '\r' || p[n] == '\n') n++;
        return n;
    }
    if (c == ' ' || c == '\t' || c == '\v' || c == '\f') {
        while (p[n] == ' ' || p[n] == '\t' || p[n] == '\v' || p[n] == '\f') n++;
        return n;
    }
    if (c >= '0' && c <= '9') {
        while (p[n] >= '0' && p[n] <= '9' && n < 3) n++;
        return n;
    }
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
        while ((p[n] >= 'a' && p[n] <= 'z') || (p[n] >= 'A' && p[n] <= 'Z')) n++;
        int cl = contraction_len(p + n);
        if (cl > 0) n += (size_t)cl;
        return n;
    }
    if (c >= 0x80) {
        while (((unsigned char)p[n]) >= 0x80) n += utf8_len_of((unsigned char)p[n]);
        return n;
    }
    while (p[n] >= 0x21 && p[n] <= 0x7E &&
           !((p[n] >= 'a' && p[n] <= 'z') || (p[n] >= 'A' && p[n] <= 'Z') ||
             (p[n] >= '0' && p[n] <= '9')))
        n++;
    return n > 0 ? n : 1;
}

static void bpe_apply(const tokenizer_t *tok, uint32_t *t, int *n_inout) {
    int n = *n_inout;
    while (n >= 2) {
        int best = -1, best_rank = INT_MAX;
        uint32_t ba = 0, bb = 0, bm = 0;
        for (int i = 0; i < n - 1; i++) {
            uint32_t merged;
            int rank = mergemap_get(&tok->merges, t[i], t[i + 1], &merged);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best = i;
                ba = t[i];
                bb = t[i + 1];
                bm = merged;
            }
        }
        if (best < 0) break;
        int w = 0;
        for (int i = 0; i < n; i++) {
            if (i < n - 1 && t[i] == ba && t[i + 1] == bb) {
                t[w++] = bm;
                i++;
            } else {
                t[w++] = t[i];
            }
        }
        n = w;
    }
    *n_inout = n;
}

size_t tokenizer_encode(tokenizer_t *tok, const char *text, uint32_t *out_tokens, size_t max_tokens) {
    if (!tok || !text || !out_tokens || max_tokens == 0) return 0;
    size_t written = 0;
    uint32_t tmp[8192];
    const char *p = text;
    while (*p && written < max_tokens) {
        size_t seg = segment_len_at(p);
        if (seg == 0) seg = 1;
        int n = 0;
        for (size_t i = 0; i < seg && p[i] && n < 8192; i++) {
            int tid = tok->byte_to_token[(unsigned char)p[i]];
            tmp[n++] = (tid >= 0) ? (uint32_t)tid : 0;
        }
        if (n >= 2) bpe_apply(tok, tmp, &n);
        for (int i = 0; i < n && written < max_tokens; i++) out_tokens[written++] = tmp[i];
        p += seg;
    }
    return written;
}

char *tokenizer_decode(tokenizer_t *tok, const uint32_t *tokens, size_t n_tokens) {
    if (!tok) { char *e = (char *)malloc(1); e[0] = 0; return e; }
    size_t total = 0;
    for (size_t i = 0; i < n_tokens; i++) {
        uint32_t id = tokens[i];
        if (id < (uint32_t)tok->vocab_size && tok->vocab[id].text)
            total += (tok->vocab[id].byte_value >= 0) ? 1 : tok->vocab[id].len;
    }
    char *r = (char *)malloc(total + 1);
    char *o = r;
    for (size_t i = 0; i < n_tokens; i++) {
        uint32_t id = tokens[i];
        if (id >= (uint32_t)tok->vocab_size || !tok->vocab[id].text) continue;
        if (tok->vocab[id].byte_value >= 0) {
            *o++ = (char)tok->vocab[id].byte_value;
        } else {
            memcpy(o, tok->vocab[id].text, tok->vocab[id].len);
            o += tok->vocab[id].len;
        }
    }
    *o = 0;
    return r;
}

int tokenizer_eos_id(const tokenizer_t *tok) {
    return tok ? tok->eos_token_id : -1;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++) free(tok->vocab[i].text);
    free(tok->vocab);
    if (tok->text2id.slots) {
        for (size_t i = 0; i < tok->text2id.cap; i++) free(tok->text2id.slots[i].key);
        free(tok->text2id.slots);
    }
    free(tok->merges.slots);
    free(tok);
}
