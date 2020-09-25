
shell_cmd": "g++ -Wall -std=c++14 -I ./3rdparty bpnn.cpp network.cpp functional.cpp loader.cpp -o bpnn.exe
# && start cmd /c \"\"${file_path}/${file_base_name}\" & pause\""
start cmd /c & ./bpnn_network.exe 