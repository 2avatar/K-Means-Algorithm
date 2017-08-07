#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Header.h"

__global__ void fillDistancePointFromPoint(float *distanceArray, Point *points, int numOfThreadsPerBlock, int numOfPoints, int iteration) {

	int index = threadIdx.x + (blockIdx.x * numOfThreadsPerBlock);

	if (index < numOfPoints) {

		float x = points[index].x - points[iteration].x;
		float y = points[index].y - points[iteration].y;

		distanceArray[index] = x*x + y*y;

	}
}

__global__ void fillPointIndexArray(Group *groups, Point *points, int *pointIndex, int numOfPoints, int numOfKlusters, int numOfThreadsPerBlock) {

	int index = threadIdx.x + (blockIdx.x * numOfThreadsPerBlock);
	float closestDistance, tempClosestDistance;
	float x, y;

	if (index < numOfPoints) {

		// assume first kluster to be closest
		pointIndex[index] = 0;
		x = groups[0].kluster.point.x - points[index].x;
		y = groups[0].kluster.point.y - points[index].y;

		closestDistance = x*x + y*y;

		// check other klusters to be closest
		for (int j = 0; j < numOfKlusters; j++) {

			x = groups[j].kluster.point.x - points[index].x;
			y = groups[j].kluster.point.y - points[index].y;

			tempClosestDistance = x*x + y*y;

			if (tempClosestDistance < closestDistance) {
				// new closest kluster and distance
				closestDistance = tempClosestDistance;
				pointIndex[index] = j;
			}
		}
	}

}

cudaError_t findGroupDiameterWithCuda(Group *group) {

	cudaError_t cudaStatus;
	float *dev_distanceArray;
	float maxDiameterArray;
	float *h_distanceArray;
	Point *dev_points;
	int numOfThreadsPerBlock = 500;
	int numOfPoints = group->numOfPoints;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	h_distanceArray = (float*)malloc(numOfPoints * sizeof(float));

	cudaStatus = cudaMalloc((void**)&dev_distanceArray, numOfPoints * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_points, numOfPoints * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// copy data from host
	cudaStatus = cudaMemcpy(dev_points, group->points, numOfPoints * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// + 1 in case of 3.2 = 3
	int numOfBlocks = ((int)(numOfPoints / numOfThreadsPerBlock) + 1);
	group->klusterDiameter = 0;
	for (int i = 0; i < numOfPoints; i++) {

		
		fillDistancePointFromPoint << <numOfBlocks, numOfThreadsPerBlock >> > (dev_distanceArray, dev_points, numOfThreadsPerBlock, numOfPoints, i);

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(h_distanceArray, dev_distanceArray, numOfPoints * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		for (int j = 0; j < numOfPoints; j++) {
			if (h_distanceArray[j] > group->klusterDiameter) {
				group->klusterDiameter = h_distanceArray[j];
			}
		}
	}

	group->klusterDiameter = sqrtf(group->klusterDiameter);
	//printf("%.3f \n", group->klusterDiameter);

Error:

	free(h_distanceArray);
	cudaFree(dev_points);
	cudaFree(dev_distanceArray);

	return cudaStatus;
}

cudaError_t groupPointsToKlustersCudaHelper(Group *groups, Point *points, int *pointIndex, int numOfPoints, int numOfKlusters) {

	Group *dev_groups;
	Point *dev_points;
	int *dev_pointIndex;
	cudaError_t cudaStatus;
	int numOfBlocks;
	int numOfThreadsPerBlock = 500;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_points, numOfPoints * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pointIndex, numOfPoints * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_groups, numOfKlusters * sizeof(Group));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// copy data from host
	cudaStatus = cudaMemcpy(dev_groups, groups, numOfKlusters * sizeof(Group), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!1");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_points, points, numOfPoints * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!2");
		goto Error;
	}

	// + 1 in case of 3.2 = 3
	numOfBlocks = ((int)(numOfPoints / numOfThreadsPerBlock) + 1);

	fillPointIndexArray << <numOfBlocks, numOfThreadsPerBlock >> >(dev_groups, dev_points, dev_pointIndex, numOfPoints, numOfKlusters, numOfThreadsPerBlock);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(pointIndex, dev_pointIndex, numOfPoints * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!3");
		goto Error;
	}

	// free allocations
Error:
	cudaFree(dev_points);
	cudaFree(dev_pointIndex);
	cudaFree(dev_groups);

	return cudaStatus;

}