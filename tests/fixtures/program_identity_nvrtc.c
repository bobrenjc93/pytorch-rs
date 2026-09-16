/* Test-only NVRTC interposer. Production still loads the real compiler ABI. */
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *Program;
enum { VERSION, CREATE, COMPILE, PTX, DESTROY, VM_SOURCE, FORWARD_ERROR, COUNTERS };
enum { FAIL_VERSION = 1, FAIL_CREATE, FAIL_COMPILE, FAIL_MODULE_LOAD };
static int phase;
static unsigned counts[COUNTERS];
static void *real_library;
static const char compile_log[] = "program identity injected compile failure";

void test_program_identity_reset(int selected) {
    phase = selected;
    memset(counts, 0, sizeof(counts));
}

unsigned test_program_identity_count(int index) {
    return index >= 0 && index < COUNTERS ? counts[index] : 0;
}

static void *symbol(const char *name) {
    if (!real_library) {
        const char *path = getenv("TEST_PROGRAM_IDENTITY_REAL_NVRTC");
        if (path) real_library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    }
    void *result = real_library ? dlsym(real_library, name) : NULL;
    if (!result) {
        const char *error = dlerror();
        fprintf(stderr, "test NVRTC forwarding failed for %s: %s\n", name,
                error ? error : "TEST_PROGRAM_IDENTITY_REAL_NVRTC is unset");
        counts[FORWARD_ERROR]++;
    }
    return result;
}

int nvrtcVersion(int *major, int *minor) {
    counts[VERSION]++;
    if (phase == FAIL_VERSION) return 7;
    int (*real)(int *, int *) = symbol("nvrtcVersion");
    return real ? real(major, minor) : 11;
}

int nvrtcCreateProgram(Program *program, const char *source, const char *name,
                       int headers, const char *const *contents, const char *const *names) {
    counts[CREATE]++;
    if (strstr(source, "switch (opcode)")) counts[VM_SOURCE]++;
    if (phase == FAIL_CREATE) return 2;
    int (*real)(Program *, const char *, const char *, int,
                const char *const *, const char *const *) = symbol("nvrtcCreateProgram");
    return real ? real(program, source, name, headers, contents, names) : 11;
}

int nvrtcCompileProgram(Program program, int count, const char *const *options) {
    counts[COMPILE]++;
    if (phase == FAIL_COMPILE) return 6;
    int (*real)(Program, int, const char *const *) = symbol("nvrtcCompileProgram");
    return real ? real(program, count, options) : 11;
}

int nvrtcGetProgramLogSize(Program program, size_t *size) {
    if (phase == FAIL_COMPILE) {
        *size = sizeof(compile_log);
        return 0;
    }
    int (*real)(Program, size_t *) = symbol("nvrtcGetProgramLogSize");
    return real ? real(program, size) : 11;
}

int nvrtcGetProgramLog(Program program, char *log) {
    if (phase == FAIL_COMPILE) {
        memcpy(log, compile_log, sizeof(compile_log));
        return 0;
    }
    int (*real)(Program, char *) = symbol("nvrtcGetProgramLog");
    return real ? real(program, log) : 11;
}

int nvrtcGetPTXSize(Program program, size_t *size) {
    int (*real)(Program, size_t *) = symbol("nvrtcGetPTXSize");
    return real ? real(program, size) : 11;
}

int nvrtcGetPTX(Program program, char *ptx) {
    counts[PTX]++;
    int (*real)(Program, char *) = symbol("nvrtcGetPTX");
    int status = real ? real(program, ptx) : 11;
    /* Retain the real buffer size/NUL terminator, but make PTX parsing fail. */
    if (status == 0 && phase == FAIL_MODULE_LOAD) ptx[0] = '!';
    /* A valid selected executor with a missing provider symbol must fail at
       linking, after NVRTC has successfully produced its real PTX. */
    if (status == 0 && phase == 5) {
        char *symbol_name = ptx;
        while ((symbol_name = strstr(symbol_name, "torch_rs_erf")) != NULL) {
            *symbol_name = 'x';
            symbol_name++;
        }
    }
    return status;
}

int nvrtcDestroyProgram(Program *program) {
    counts[DESTROY]++;
    int (*real)(Program *) = symbol("nvrtcDestroyProgram");
    return real ? real(program) : 11;
}
