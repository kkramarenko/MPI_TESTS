#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <malloc.h>
#include <time.h>
#include <math.h>

#define debug_print(fmt, ...) \
            do { if (DEBUG) fprintf(stdout, fmt, ##__VA_ARGS__); } while (0)

unsigned int least_sign_bit(unsigned int number)
{
    unsigned int lsb = 1;

    if (number == 0) return 0;

    while (!(number&1)){
	number >>= 1;
	lsb <<= 1;
    }

    return lsb;
}

int* bruck_allgather(unsigned int rank,unsigned int node_count, int *block, int block_size, int *buffer_size)
{
    unsigned int remain, lsb, steps;
    int *buffer = NULL;
    int *buffer_recv = NULL;
    double tmp;
    int offset = 0;
    int index, step;
    int flag;
    int recv_size = 0;
    int send_rank, recv_rank;
    MPI_Status status[2];
    MPI_Request req[2];

    tmp = log2(node_count);
    steps = (int) tmp;
    debug_print("rank %u - step = %u block_size = %d\n", rank, steps, block_size);
    tmp = pow(2.0, (int)tmp);
    remain = node_count - ((int) tmp);
    debug_print("rank %u - remain = %u\n", rank, remain);
    lsb = least_sign_bit(remain);
    debug_print("rank %u - lsb = %u\n", rank, lsb);
    remain = remain - lsb;
    debug_print("rank %u - remain = %u\n", rank, remain);

    debug_print("rank - %d buffer = %p\n",rank, buffer);
    buffer = malloc(block_size * sizeof(int));
    debug_print("rank - %d buffer= %p\n",rank, buffer);
    memcpy(buffer, block, block_size * sizeof(int));

    debug_print("rank - %d buffer = ", rank);
    for(index = 0; index < block_size; index++)
	debug_print("%d ", buffer[index]);
    debug_print("\n");

    *buffer_size = block_size;
    for (step = 0; step < steps; step++){
	if (remain & (1 << step)) {
	    buffer = realloc(buffer, (*buffer_size + 1) * sizeof(int));
	    for (index = *buffer_size; index >= 1; index --) {
		buffer[index] = buffer[index - 1];
	    }
	    buffer[0] = offset;
	}
	
	send_rank = (rank - (1 << step)) % node_count;
	recv_rank = (rank + (1 << step)) % node_count;
	MPI_Send(buffer, *buffer_size, MPI_INT, send_rank, 0, MPI_COMM_WORLD);
	//while(!flag){
	//    MPI_Iprobe(recv_rank, 0, MPI_COMM_WORLD, &flag, &status[0]);
	//}
	MPI_Probe(recv_rank, 0, MPI_COMM_WORLD, &status[0]);
	MPI_Get_count(&status[0], MPI_INT, &recv_size);
	buffer_recv = malloc(recv_size * sizeof(int));
	MPI_Recv(buffer_recv, recv_size, MPI_INT, recv_rank, 0, MPI_COMM_WORLD, &status[0]);
	//MPI_Waitall(2, req, status);
	debug_print("rank - %d error0 = %d error1 = %d\n", rank, status[0].MPI_ERROR, status[1].MPI_ERROR);
	debug_print("rank - %d recv_buffer = ", rank);
	for (index = 0; index < recv_size; index++)
	    debug_print("%d ", buffer_recv[index]);
	debug_print("\n");
	
	if (lsb & (1 << step)) {
	    offset = *buffer_size;
	}

	if (remain & (1 << step)) {
	    offset = buffer_recv[0] + *buffer_size;
	    recv_size = recv_size - 1;
	    buffer_recv = buffer_recv + 1;
	}

	buffer = realloc(buffer, (*buffer_size + recv_size) * sizeof(int));
	memcpy(buffer + (*buffer_size), buffer_recv, recv_size * sizeof(int));
	*buffer_size = *buffer_size + recv_size;
	free(buffer_recv);
    }
    
    if (offset) {
	debug_print("rank - %d offset<>0\n", rank);
	send_rank = (rank - (1 << steps)) % node_count;
	recv_rank = (rank + (1 << steps)) % node_count;
	MPI_Send(buffer, offset, MPI_INT, send_rank, 0, MPI_COMM_WORLD);
	MPI_Probe(recv_rank, 0, MPI_COMM_WORLD, &status[0]);
	MPI_Get_count(&status[0], MPI_INT, &recv_size);
	buffer_recv = malloc(recv_size * sizeof(int));
	MPI_Recv(buffer_recv, recv_size,MPI_INT, recv_rank, 0, MPI_COMM_WORLD, &status[0]);
	buffer = realloc(buffer, (*buffer_size + recv_size) * sizeof(int));
	memcpy(buffer + (*buffer_size), buffer_recv, recv_size * sizeof(int));
	*buffer_size = *buffer_size + recv_size;
	free(buffer_recv);
    }

    return buffer;
}

int main(int argc, char **argv)
{
    int index;
    unsigned int rank, size;
    unsigned int seed = 0;
    int block_size = 0;
    int  *block  = NULL;
    int  *buffer = NULL;
    int buffer_size = 0;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("I`m  node - %d of %d\n",rank, size);

    seed = ((unsigned int) time(NULL)) + rank;
    block_size = rand_r(&seed);
    block_size = block_size % 10 + 2;
    debug_print("rank %d - block size = %d\n", rank, block_size);

    block = malloc(block_size * sizeof(int));
    for(index = 0; index < block_size; index ++)
	block[index] = rank;

    printf("rank %d - block = ", rank);
    for (index = 0;index < block_size; index++)
	printf("%d ", (int) block[index]);
    printf("\n");

    debug_print("rank - %d buffer = %p\n", rank, buffer);
    buffer = bruck_allgather(rank, size, block, block_size, &buffer_size);

    printf("rank - %d rezult buffer = ", rank);
    for (index = 0; index < buffer_size; index ++)
	printf("%d ", buffer[index]);
    printf("\n");
    
    free(block);
    free(buffer);
    MPI_Finalize();
}
