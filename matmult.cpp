#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

// ---------------------------------------------------------------------------
// allocate space for empty matrix A[row][col]
// access to matrix elements possible with:
// - A[row][col]
// - A[0][row*col]

float **alloc_mat(int row, int col)
{
    float **A1, *A2;

	A1 = (float **)calloc(row, sizeof(float *));		// pointer on rows
	A2 = (float *)calloc(row*col, sizeof(float));    // all matrix elements
    for (int i = 0; i < row; i++)
        A1[i] = A2 + i*col;

    return A1;
}

// ---------------------------------------------------------------------------
// random initialisation of matrix with values [0..9]

void init_mat(float **A, int row, int col)
{
    for (int i = 0; i < row*col; i++)
		A[0][i] = (float)(rand() % 10);
}

// ---------------------------------------------------------------------------
// DEBUG FUNCTION: printout of all matrix elements

void print_mat(float **A, int row, int col, char *tag)
{
    int i, j;

    printf("Matrix %s:\n", tag);
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++) 
            printf("%6.1f   ", A[i][j]);
        printf("\n"); 
    }
}



// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {   
    int numtasks;
    int count;
    int taskid;
    int numworkers;
    int source;
    int dest;
    int mtype;
    int rows;
    int averow, extra, offset;
    double starttime, endtime;
	float **A, **B, **C;
    float **At, **Bt, **Ct;	// matrices
    int d1t, d2t, d3t;         // dimensions of matrices
    int i, j, k, rc;			// loop variables


    MPI_Status status;
    //MPI_Pcontrol(APPLICATION, string);
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

   if (argc != 4) {
        printf ("Matrix multiplication: C = A x B\n");
        printf ("Usage: %s <NumRowA> <NumColA> <NumColB>\n", argv[0]); 
        //MPI_Abort(MPI_COMM_WORLD,rc);
        //MPI_Finalize();
        return 0;
    }
    
    if (numtasks < 2 ) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD,0);
    exit(1);
    }

     /* read user input */
    d1 = atoi(argv[1]);     // rows of A and C  d1
    d2 = atoi(argv[2]);     // cols of A and rows of B  d2
    d3 = atoi(argv[3]); 

    d1=d1t;
    d2=d2t;
    d3=d3t;

     

        // cols of B and C d3

    printf("Matrix sizes C[%d][%d] = A[%d][%d] x B[%d][%d]\n", d1, d3, d1, d2, d2, d3);

    /* prepare matrices */
   
    
    numworkers = numtasks-1;
     /* Code für den Manager */
    if (taskid == MASTER) {
        printf("matrix multiplikation with MPI\n");
        
            
            A = alloc_mat(d1, d2); init_mat(A, d1, d2); 
            B = alloc_mat(d2, d3); init_mat(B, d2, d3);
            C = alloc_mat(d1, d3);
                
           

             /* Matrizen versenden */
            //numworkers= 2 ;
            averow = d1/numworkers;
            extra = d1%numworkers;
            offset = 0;
            mtype = FROM_MASTER;
            
            starttime=MPI_Wtime();
            
            for (dest=1;dest<numworkers;dest++) {
                rows = (dest <= extra) ? averow+1 :averow;
                printf("Sending %drows to task %doffset=%d\n",rows,dest,offset);
                MPI_Send(&offset, 1, MPI_INT,dest,mtype, MPI_COMM_WORLD);
                MPI_Send(&rows, 1, MPI_INT,dest,mtype, MPI_COMM_WORLD);
                MPI_Send(&A[offset][0],rows*d2, MPI_DOUBLE,dest,mtype, MPI_COMM_WORLD);
                MPI_Send(&B, d2*d3, MPI_DOUBLE,dest,mtype, MPI_COMM_WORLD);
                offset =offset +rows;
            }

             /* Ergebnisse empfangen */
            mtype = FROM_WORKER;
            
            for (i=1; i<=numworkers; i++) {
                source = i;
                MPI_Recv(&offset, 1, MPI_INT,source,mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&rows, 1, MPI_INT,source,mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&C[offset][0],rows*d3, MPI_DOUBLE,source,mtype,MPI_COMM_WORLD, &status);
                printf("Received results from task %d\n",source);
            }

            printf("result :\n");
                for (i=0; i<d1; i++) {
                        printf("\n");
                    for (j=0; j<d3; j++) {
                        printf("%6.2f ", C[i][j]);
                    }
                }
            
            endtime=MPI_Wtime();
            printf("\nIt took %fseconds.\n",endtime-starttime);
    }       
    
    /* Code für die Arbeiter */
    
    if (taskid > MASTER) {
        mtype = FROM_MASTER;
        
        MPI_Recv(&offset, 1, MPI_INT, MASTER,mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&d1, 1, MPI_INT, MASTER,mtype, MPI_COMM_WORLD, &status);
        A = alloc_mat(rows, d2); 
        B = alloc_mat(d2, d3);
        C = alloc_mat(rows, d3);
        MPI_Recv(&A,rows*d2, MPI_DOUBLE, MASTER,mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&B, d2*d3, MPI_DOUBLE, MASTER,mtype, MPI_COMM_WORLD, &status);

    /* print user instruction */
    

   	// no initialisation of C, because it gets filled by matmult

    /* serial version of matmult */
        printf("Perform matrix multiplication...\n");
        for (i = 0; i < d1; i++)
            for (j = 0; j < d3; j++)
                //C[i][j]= 0.0;
                for (k = 0; k < d2; k++)
                C[i][j] += A[i][k] * B[k][j];
        
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER,mtype, MPI_COMM_WORLD);
        MPI_Send(&d1, 1, MPI_INT, MASTER,mtype, MPI_COMM_WORLD);
        MPI_Send(&C,rows*d3, MPI_DOUBLE, MASTER,mtype, MPI_COMM_WORLD);
    
    }
    
    MPI_Finalize();


    /* test output 
    print_mat(A, d1, d2, "A"); 
    print_mat(B, d2, d3, "B"); 
    print_mat(C, d1, d3, "C"); */

    printf ("\nDone.\n");


    //return 0;
}
