// Linux link stubs for Apple Metal / Objective-C symbols declared in `src/metal.gsa`.
// Never executed on Linux (metalAvailable() returns false when MTLCreateSystemDefaultDevice returns NULL).

#include <stddef.h>

void *MTLCreateSystemDefaultDevice(void) { return NULL; }
void *objc_getClass(const char *name) { (void)name; return NULL; }
void *sel_registerName(const char *name) { (void)name; return NULL; }
void *objc_autoreleasePoolPush(void) { return NULL; }
void objc_autoreleasePoolPop(void *pool) { (void)pool; }
void *objc_msgSend(void *obj, void *sel, ...) { (void)obj; (void)sel; return NULL; }
