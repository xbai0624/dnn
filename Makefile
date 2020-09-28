# ################################################ 
#      a simplified multipurpose makefile
# ################################################
#
CC		:= g++
Compile_Flags	:= -Wall -Wextra  -g -std=c++11 -O2 -pthread

# these options for generating library
Target		:= main

INCS		:= -I./include

SRCS		:= $(wildcard src/*.cpp)
OBJS		:= $(addprefix obj/, $(patsubst %.cpp, %.o, $(notdir ${SRCS})))

objects = ${OBJS} # static patterns

all: ${Target}

# generate executable
${Target}: ${OBJS}
	${CC} ${Compile_Flags} -o $@ $^ ${INCS}

# static patterns
${objects}: obj/%.o: src/%.cpp
	${CC} ${Compile_Flags} -o $@ -c $< ${INCS}

clean:
	@rm -rf obj/* ${Target}
	@rm -rf ${OBJS}

test:
	@echo ${OBJS}
