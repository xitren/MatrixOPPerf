#include <stdio.h>  /* Для функции printf() */
#include <stdlib.h> /* Для маркеров статуса */

void
helloFromCPU(void)
{ /* Эта функция работает на хосте */
    printf("Hello World from CPU!\n");
}

__global__ void
helloFromGPU()
{ /* Это ядро запускается на устройстве */
    printf("Hello World from GPU!\n");
}

int
main(int argc, char** argv)
{
    helloFromCPU();           /* Вызов с хоста */
    helloFromGPU<<<1, 1>>>(); /* Запуск с хоста */
    cudaDeviceReset();        /* Уборка на устройстве */
    return (EXIT_SUCCESS);
}