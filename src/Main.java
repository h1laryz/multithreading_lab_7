import mpi.*;

import java.util.Arrays;

public class Main
{
    private static final int NUMBER_ROWS_A = 10;
    private static final int NUMBER_COLUMNS_A = 10;
    private static final int NUMBER_COLUMNS_B = 10;
    private static final int MASTER = 0;

    public static void main(String[] args)
    {
        double[][] matrix_a = new double[NUMBER_ROWS_A][NUMBER_COLUMNS_A];
        double[][] matrix_b = new double[NUMBER_COLUMNS_A][NUMBER_COLUMNS_B];
        double[][] result_matrix = new double[NUMBER_ROWS_A][NUMBER_COLUMNS_B];

        MPI.Init(args);
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        if (size < 2)
        {
            System.out.println("Need at least 2 MPI processes.\n");
            MPI.COMM_WORLD.Abort(1);
        }
        if(rank == MASTER)
        {
            for (int i = 0; i < NUMBER_ROWS_A; i++)
            {
                for (int j = 0; j < NUMBER_COLUMNS_A; j++)
                {
                    matrix_a[i][j] = 1;
                }
            }
            for (int i = 0; i < NUMBER_COLUMNS_A; i++)
            {
                for (int j = 0; j < NUMBER_COLUMNS_B; j++)
                {
                    matrix_b[i][j] = 1;
                }
            }
        }

        int[] offset = new int[size];
        int[] rows = new int[size];

        int amount_for_process = NUMBER_ROWS_A / size;
        int extra = NUMBER_ROWS_A % size;

        for (int i = 0; i < size; i++)
        {
            rows[i] = i < extra ? amount_for_process + 1 : amount_for_process;
            offset[i] = i == MASTER ? 0 : offset[i-1] + rows[i-1];
        }

        int local_matrix_a_rows = rows[rank];
        double[][] local_matrix_a = new double[local_matrix_a_rows][NUMBER_COLUMNS_A];

        MPI.COMM_WORLD.Scatterv(matrix_a, 0, rows, offset, MPI.OBJECT,
                local_matrix_a, 0, local_matrix_a_rows, MPI.OBJECT, MASTER
        );

        MPI.COMM_WORLD.Bcast(matrix_b, 0, NUMBER_COLUMNS_A, MPI.OBJECT, MASTER);

        double[][] local_result_matrix_rows = new double[local_matrix_a_rows][NUMBER_COLUMNS_B];
        for (int k = 0; k < NUMBER_COLUMNS_B; k++)
        {
            for (int i = 0; i < local_matrix_a_rows; i++)
            {
                for (int j = 0; j < NUMBER_COLUMNS_A; j++)
                {
                    local_result_matrix_rows[i][k] += local_matrix_a[i][j] * matrix_b[j][k];
                }
            }
        }

        MPI.COMM_WORLD.Gatherv(local_result_matrix_rows, 0, local_matrix_a_rows, MPI.OBJECT,
                result_matrix, 0, rows, offset, MPI.OBJECT,
                MASTER
        );

        if (rank == MASTER)
        {
            for(int i = 0; i < NUMBER_ROWS_A; i++)
            {
                for (int j = 0; j < NUMBER_COLUMNS_B; j++)
                {
                    System.out.print(result_matrix[i][j] + " ");
                }
                System.out.print('\n');
            }
        }

        MPI.Finalize();
    }
}