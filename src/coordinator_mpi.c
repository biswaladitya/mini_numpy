#include <mpi.h>

#include "coordinator.h"
#include <string.h>

#define READY 0
#define NEW_TASK 1
#define TERMINATE -1


int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Error: not enough arguments\n");
    printf("Usage: %s [path_to_task_list]\n", argv[0]);
    return -1;
  }

  int num_tasks;
  task_t **tasks;
  if (read_tasks(argv[1], &num_tasks, &tasks))
    return -1;

  MPI_Init(&argc, &argv); 

  int procID, totalProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procID);
  if (procID == 0) {  // manager node
    // Manager node
    int32_t nextTask = 0;
    MPI_Status status;
    int32_t message;

    while(nextTask<num_tasks){
      MPI_Recv(&message, 1, MPI_INT32_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      int source = status.MPI_SOURCE;
      message = nextTask;
      MPI_Send(&message, 1, MPI_INT32_T, source, 0, MPI_COMM_WORLD);
      nextTask++;
    }

    for (int i = 1; i < totalProcs; i++) { 
      MPI_Recv(&message, 1, MPI_INT32_T, MPI_ANY_SOURCE, READY, MPI_COMM_WORLD, &status); 
      int source = status.MPI_SOURCE; 
      message = TERMINATE;
      MPI_Send(&message, 1, MPI_INT32_T, source, 0, MPI_COMM_WORLD);
    }
  } else { //worker node

    int32_t message;

    while(true) {
      message = READY;
      MPI_Send(&message, 1, MPI_INT32_T, 0, READY, MPI_COMM_WORLD);
      MPI_Recv(&message, 1, MPI_INT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int taskIndex = message;
      if (taskIndex == TERMINATE) {
          break;
      }
      execute_task(tasks[taskIndex]);
    }
  }
  MPI_Finalize();
}
