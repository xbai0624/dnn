Target		:= main
CC		:= g++
CXXFLAGS	:= -std=c++11 -Wall -O2 -g

ROOT_INC	:= -I${ROOTSYS}/include
INC		:= -Iinclude ${ROOT_INC}

ROOT_LIB	:= $(shell root-config --glibs) -lMathMore

SRCS		:= $(wildcard src/*.cpp)
OBJS		:= $(addprefix obj/, $(patsubst %.cpp, %.o, $(notdir ${SRCS})))

${Target}:${OBJS}
	${CC} ${CXXFLAGS} -o $@ $^ ${INC} ${ROOT_LIB} 

obj/%.o: src/%.cpp
	${CC} ${CXXFLAGS} -o $@ -c $< ${INC} ${ROOT_LIB}

clean:
	@rm -rf ${Target} ${OBJS}
test:
	echo ${INC}
