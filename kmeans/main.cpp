#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Header.h"

int main(int argc, char *argv[]) {

	float data[SIZE_OF_DATA];
	float numOfPoints, maxKlusters, maxIterations, maxQualityMeasure;
	float pointX, pointY;
	double trash;
	Point *points;
	Group *groups;
	float **distancePointsFromPointMatrix;
	double startTime, endTime;
	int myid, numprocs;
	int numOfKlusters = 0;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Status status;

	if (numprocs != NUM_OF_CPU) {
		printf("Incorrect number of threads, should be: %d \n", NUM_OF_CPU);
		MPI_Finalize();
		return 0;
	}

	if (myid == MASTER){

		char *inputFileName = "C:/Users/eviat/Documents/Visual Studio 2015/Projects/kmeans - paralleled with algorithm v2/kmeans/input.txt";
		char *outputFileName = "C:/Users/eviat/Documents/Visual Studio 2015/Projects/kmeans - paralleled with algorithm v2/kmeans/output.txt";
		int firstSlaveNumOfKlusters, secondSlaveNumOfKlusters, terminate = -1;
		float firstSlaveQualityMeasure, secondSlaveQualityMeasure, masterQualityMeasure=0;
		int bestNumOfKlusters;
	
		// load data from file
		FILE *f = fopen(inputFileName, "r");
		if (f == NULL) {
			printf("Failed opening the file. Exiting!\n");
			MPI_Finalize();
			return 0;
		}
		fscanf(f, "%f,%f,%f,%f", &numOfPoints, &maxKlusters, &maxIterations, &maxQualityMeasure);

		data[0] = numOfPoints;
		data[1] = maxKlusters;
		data[2] = maxIterations;
		data[3] = maxQualityMeasure;

		// send initial data
		MPI_Send(data, SIZE_OF_DATA, MPI_FLOAT, FIRST_SLAVE, BROADCAST_NUM, MPI_COMM_WORLD);
		MPI_Send(data, SIZE_OF_DATA, MPI_FLOAT, SECOND_SLAVE, BROADCAST_NUM, MPI_COMM_WORLD);

		points = (Point*)malloc(numOfPoints * sizeof(Point));

		// load points from file
		for (int i = 0; i < numOfPoints; i++) {
			fscanf(f, "%d,%f,%f", &trash, &pointX, &pointY);
			points[i].x = pointX;
			points[i].y = pointY;
			// send points to slaves
			MPI_Send(&points[i], sizeof(Point), MPI_FLOAT, FIRST_SLAVE, BROADCAST_NUM, MPI_COMM_WORLD);
			MPI_Send(&points[i], sizeof(Point), MPI_FLOAT, SECOND_SLAVE, BROADCAST_NUM, MPI_COMM_WORLD);
		}

		fclose(f);

		startTime = omp_get_wtime();

		//1. set k=2
		numOfKlusters = 2;

		for (numOfKlusters; numOfKlusters < maxKlusters; numOfKlusters += NUM_OF_CPU) {

			firstSlaveNumOfKlusters = numOfKlusters + 1;
			secondSlaveNumOfKlusters = numOfKlusters + 2;

			// send iteration with num of klusters to slaves
			MPI_Send(&firstSlaveNumOfKlusters, SIZE_OF_INT, MPI_INT, FIRST_SLAVE, BROADCAST_NUM, MPI_COMM_WORLD);
			MPI_Send(&secondSlaveNumOfKlusters, SIZE_OF_INT, MPI_INT, SECOND_SLAVE, BROADCAST_NUM, MPI_COMM_WORLD);

			groups = (Group*)malloc(numOfKlusters * sizeof(Group));
			// do the algorithm as a master
			masterQualityMeasure = kMeansAlgorithm(numOfKlusters, numOfPoints, maxIterations, points, groups);
			
			// get quality measure from slaves
			MPI_Recv(&firstSlaveQualityMeasure, SIZE_OF_FLOAT, MPI_FLOAT, FIRST_SLAVE, BROADCAST_NUM, MPI_COMM_WORLD, &status);
			MPI_Recv(&secondSlaveQualityMeasure, SIZE_OF_FLOAT, MPI_FLOAT, SECOND_SLAVE, BROADCAST_NUM, MPI_COMM_WORLD, &status);
			
			printf("Number of Klusters: %d , Quality Measure: %.3f \n", numOfKlusters, masterQualityMeasure);
			printf("Number of Klusters: %d , Quality Measure: %.3f \n", firstSlaveNumOfKlusters, firstSlaveQualityMeasure);
			printf("Number of Klusters: %d , Quality Measure: %.3f \n", secondSlaveNumOfKlusters, secondSlaveQualityMeasure);

			//7. Evaluate the quality of the clusters 
			if (masterQualityMeasure > maxQualityMeasure) {
				bestNumOfKlusters = numOfKlusters;
				break;
			}
			else if (firstSlaveQualityMeasure > maxQualityMeasure) {
				bestNumOfKlusters = firstSlaveNumOfKlusters;
				break;
			}
			else if (secondSlaveQualityMeasure > maxQualityMeasure) {
				bestNumOfKlusters = secondSlaveNumOfKlusters;
				break;
			}

		// free before next iteration
		free(groups);
		
		}

		// send termination flag to slaves
		MPI_Send(&terminate, SIZE_OF_INT, MPI_INT, FIRST_SLAVE, BROADCAST_NUM, MPI_COMM_WORLD);
		MPI_Send(&terminate, SIZE_OF_INT, MPI_INT, SECOND_SLAVE, BROADCAST_NUM, MPI_COMM_WORLD);

		// do last best iteration
		groups = (Group*)malloc(bestNumOfKlusters * sizeof(Group));
		masterQualityMeasure = kMeansAlgorithm(bestNumOfKlusters, numOfPoints, maxIterations, points, groups);
		printf("BEST: Number of Klusters: %d , Quality Measure: %.3f \n", bestNumOfKlusters, masterQualityMeasure);

		// write data to output file
		f = fopen(outputFileName, "w");
		if (f == NULL) {
			printf("Failed opening the file. Exiting!\n");
			MPI_Finalize();
			return 0;
		}

		fprintf(f, "%d %f \n", bestNumOfKlusters, masterQualityMeasure);
		for (int i = 0; i < bestNumOfKlusters; i++) 
			fprintf(f, "x = %.3f, y = %.3f \n", groups[i].kluster.point.x, groups[i].kluster.point.y);
		
		fclose(f);

		free(groups);

		endTime = omp_get_wtime();
		printf("time = %g \n", endTime - startTime);

	}
	else {

		float slaveQualityMeasure = 10;

		// recieve data from master
		MPI_Recv(data, SIZE_OF_DATA, MPI_FLOAT, MASTER, BROADCAST_NUM, MPI_COMM_WORLD, &status);
		
		numOfPoints = data[0];
		maxKlusters = data[1];
		maxIterations = data[2];
		maxQualityMeasure = data[3];
		
		// allocate array for points
		points = (Point*)malloc(numOfPoints * sizeof(Point));
		
		// get points from master
		for (int i=0; i<numOfPoints; i++)
		MPI_Recv(&points[i], sizeof(Point), MPI_FLOAT, MASTER, BROADCAST_NUM, MPI_COMM_WORLD, &status);

		// keep on doing algorithm untill master sends termination flag
		while (numOfKlusters != TERMINATION_FLAG) {

			// get num of klusters
			MPI_Recv(&numOfKlusters, SIZE_OF_INT, MPI_INT, MASTER, BROADCAST_NUM, MPI_COMM_WORLD, &status);

			if (numOfKlusters != TERMINATION_FLAG) {
				groups = (Group*)malloc(numOfKlusters * sizeof(Group));
				// do the algorithm as a slave
				slaveQualityMeasure = kMeansAlgorithm(numOfKlusters, numOfPoints, maxIterations, points, groups);
				// send quality measure to master
				free(groups);
				MPI_Send(&slaveQualityMeasure, SIZE_OF_FLOAT, MPI_FLOAT, MASTER, BROADCAST_NUM, MPI_COMM_WORLD);
			}
		}
	}

	MPI_Finalize();
	return 0;


	//char *inputFileName = "input.txt";

	//// load data from file
	//FILE *f = fopen(inputFileName, "r");
	//if (f == NULL) {
	//	printf("Failed opening the file. Exiting!\n");
	//	return 0;
	//}
	//fscanf(f, "%f,%f,%f,%f", &numOfPoints, &maxKlusters, &maxIterations, &maxQualityMeasure);

	//points = (Point*)malloc(numOfPoints * sizeof(Point));

	//// load points from file
	//for (int i = 0; i < numOfPoints; i++) {
	//	fscanf(f, "%d,%f,%f", &trash, &pointX, &pointY);
	//	points[i].x = pointX;
	//	points[i].y = pointY;
	//}

	//fclose(f);

	//// iteration of klusters
	//numOfKlusters = 2;
	//startTime = omp_get_wtime();
	//for (numOfKlusters; numOfKlusters < maxIterations; numOfKlusters++) {
	//	Group *groups = (Group*)malloc(numOfKlusters * sizeof(Group));
	//	float qualityMeasure = kMeansAlgorithm(numOfKlusters, numOfPoints, maxIterations, points, groups);
	//	printf("Number of Klusters: %d , Quality Measure: %.3f \n", numOfKlusters, qualityMeasure);
	//	if (qualityMeasure > maxQualityMeasure)
	//		break;
	//	free(groups);
	//}

	//endTime = omp_get_wtime();
	//printf("total time= %g \n", endTime - startTime);

	//free(&groups);
	//return 0;

}


