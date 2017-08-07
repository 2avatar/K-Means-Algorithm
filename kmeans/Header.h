#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_OF_CPU 3
#define TERMINATION_FLAG -1
#define SIZE_OF_FLAG 1
#define SIZE_OF_DATA 4
#define SIZE_OF_FLOAT 1
#define SIZE_OF_INT 1
#define BROADCAST_NUM 0
#define MASTER 0
#define FIRST_SLAVE 1
#define SECOND_SLAVE 2

typedef struct {

	float x;
	float y;

}Point;

typedef struct {

	Point point;

}Kluster;

typedef struct {

	Kluster kluster;
	Point* points;
	int numOfPoints;
	float klusterDiameter;

}Group;

float kMeansAlgorithm(int numOfKlusters, float numOfPoints, float maxIterations, Point *points, Group *groups);
cudaError_t findGroupDiameterWithCuda(Group *group);
float calculateQualityMeasure(Group *groups, int numOfKlusters);
void calculateNewKlustersCenterOfMass(Group *groups, int numOfKlusters);
void groupPointsToKlusters(Kluster *klusters, Point *points, Group *groups, int numOfKlusters, int numOfPoints);
void findGroupDiamaeterWithAlgorithm(Group *group);
cudaError_t groupPointsToKlustersCudaHelper(Group *groups, Point *points, int *pointIndex, int numOfPoints, int numOfKlusters);
float calculateDistance(Point firstPoint, Point secondPoint);
void printMatrix(float **matrix, int row, int col);
void printPoints(Point *points, int numOfPoints);




