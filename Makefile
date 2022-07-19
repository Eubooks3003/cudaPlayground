include ../../../shared/CUDA.mk

ALL_CCFLAGS := $(ALL_CCFLAGS) --ptxas-options=-v

.PHONY: all clean

all: build

KERNEL_SOURCES := $(wildcard *.cu)
KERNEL_SASSES := $(KERNEL_SOURCES:.cu=.S)
KERNEL_OBJS := $(KERNEL_SOURCES:.cu=.o)
KERNEL_EXES := main.exe

build: $(KERNEL_EXES) $(KERNEL_SASSES)

%.exe: $(KERNEL_OBJS)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

%.S: %.o
	$(CUOBJDUMP) --dump-sass $< > $@

.PRECIOUS: $(KERNEL_OBJS)
%.o: %.cu Makefile
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

.PHONY: clean
clean:
	$(RM) $(KERNEL_EXES)
