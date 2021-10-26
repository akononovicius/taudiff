CC = gcc

SRC_DIR = fgn-generator-gsl

LINKER_FLAGS = -lgsl -lgslcblas
OBJ_C_FLAGS = -c -fpic
OBJ_SO_FLAGS = -shared -o

all: libfgn.so

#
# SHARED OBJECTS
#
libfgn.so: apprcirc.o spectrum.o
	${CC} ${OBJ_SO_FLAGS} libfgn.so apprcirc.o spectrum.o ${LINKER_FLAGS}

#
# COMPONENT OBJECTS
#
apprcirc.o: ${SRC_DIR}/apprcirc.c ${SRC_DIR}/apprcirc.h
	${CC} ${OBJ_C_FLAGS} ${SRC_DIR}/apprcirc.c ${LINKER_FLAGS}

spectrum.o: ${SRC_DIR}/spectrum.c ${SRC_DIR}/spectrum.h
	${CC} ${OBJ_C_FLAGS} ${SRC_DIR}/spectrum.c ${LINKER_FLAGS}

#
# GENERAL CMDS
#
clean:
	rm *.o

# vim: set noexpandtab tabstop=4 shiftwidth=4 softtabstop=-1:
