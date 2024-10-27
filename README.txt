Calculadora de producto tensorial con algoritmo paralelo en CUDA
-------------------------
Compilacion: nvcc t4.exe -o t4.cu
-------------------------
Ejecucion: ./t4.exe -M -O data.txt
donde: 
      M = {B: procesamiento solo con bloques, T: procesamiento con bloques y hebras}
      O = {S (Silent), V (Verbose)}
Se incluyen data.txt data2.txt, data3.txt y data4.txt como datos de prueba.
------------------
Maquina de prueba:

CPU: Ryzen 5 2600x
RAM: 16 gb ram
OS: Ubuntu 22.04.3
COMPILADORES: gcc 11.4.0, MPICH 4.0





