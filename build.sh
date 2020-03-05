clang++-10 -std=c++2a -O3 -flto -fno-exceptions -static -march=native -fno-rtti -Wall -Winline -Wextra --extra-warnings -Wno-unknown-cuda-version -Wno-deprecated-anon-enum-enum-conversion -Wno-deprecated-copy main.cpp -I/home/xdaimon/metann/

#https://stackoverflow.com/questions/15930755/are-lambdas-inlined-like-functions-in-c
#-Winline warns if a function marked as inline cannot be inlined


