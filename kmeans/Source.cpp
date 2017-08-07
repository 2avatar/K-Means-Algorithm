#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Header.h"

float calculateQualityMeasure(Group *groups, int numOfKlusters) {

	float qualityMeasure = 0;
	int i;

	// sum all diameters divide by distance of klusters
#pragma omp parallel reduction (+: qualityMeasure)
	{
#pragma omp for private (i)
		for (i = 0; i < numOfKlusters; i++) {
			for (int j = 0; j < numOfKlusters; j++) {
				if (i!=j)
				qualityMeasure += (groups[j].klusterDiameter) / calculateDistance(groups[i].kluster.point, groups[j].kluster.point);
			}
		}
	}

	return qualityMeasure;
}

void calculateNewKlustersCenterOfMass(Group *groups, int numOfKlusters) {

	float x;
	float y;
	int i, j;

	for (i = 0; i < numOfKlusters; i++) {

		x = 0;
		y = 0;

		// sum all x points y points
#pragma omp parallel reduction (+: x,y)
		{

#pragma omp for private (j)
			for (j = 0; j < groups[i].numOfPoints; j++) {

				x += groups[i].points[j].x;
				y += groups[i].points[j].y;

			}
		}
			// divide sum by num of points
		groups[i].kluster.point.x = (x / groups[i].numOfPoints);
		groups[i].kluster.point.y = (y / groups[i].numOfPoints);
	}

}

void groupPointsToKlusters(Kluster *klusters, Point *points, Group *groups, int numOfKlusters, int numOfPoints) {

	int closestKlusterNumber;
	int closestDistance, tempClosestDistance;
	int i = 0, j=0;
	int tid;

	// allocate memory for new points index
	int *pointIndex = (int*)malloc(numOfPoints*sizeof(int));

	// group klusters + initiate group number of points
	for (i = 0; i < numOfKlusters; i++) {
		groups[i].kluster = klusters[i];
		groups[i].numOfPoints = 0;
	}

	// group temporary points
	groupPointsToKlustersCudaHelper(groups, points, pointIndex, numOfPoints, numOfKlusters);

	//caculate groups num of points
	for (i=0; i<numOfPoints; i++)
		groups[pointIndex[i]].numOfPoints++;

	// group points
	for (i = 0; i < numOfKlusters; i++) {
		groups[i].points = (Point*)malloc(groups[i].numOfPoints*sizeof(Point));
	
		int index = 0;
		for (int j=0; j<numOfPoints; j++){
		
			if (pointIndex[j] == i){
			groups[i].points[index] = points[j];
			index++;
			}
		}
	}

	// free allocations
	free(pointIndex);

}

float calculateDistance(Point firstPoint, Point secondPoint) {

	float x = firstPoint.x - secondPoint.x;
	float y = firstPoint.y - secondPoint.y;

	return sqrtf(x*x + y*y);
}

void printMatrix(float **matrix, int row, int col) {

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			printf("%.3f ", matrix[i][j]);
		}
		printf("\n");
	}
}

void printPoints(Point *points, int numOfPoints) {

	for (int i = 0; i < numOfPoints; i++)
		printf("x = %.3f, y = %.3f \n", points[i].x, points[i].y);

}

void findGroupDiamaeterWithAlgorithm(Group *group) {

	int const numOfCornerPoints = 8;
	int numOfPoints = group->numOfPoints;
	float tempDiameter;
	float maxY = 0, maxX = 0, minY, minX;
	float maxYmaxX = 0, maxYminX, maxXmaxY = 0, maxXminY, minYmaxX = 0, minYminX, minXmaxY = 0, minXminY;
	bool firstMaxYminX = true, firstMaxXminY = true, firstMinYminX = true, firstMinXminY = true;

	Group *tempGroup;
	Point cornerPoints[numOfCornerPoints];
	Point circleCenter;
	Point *SuspeciousPoints;
	float distanceFromCircleCenter;
	int numOfSuspeciousPoints = 0, counterSuspeciousPoints = numOfCornerPoints;

	// find max y max x min y min x
	minY = group->points[0].y;
	minX = group->points[0].x;
	for (int i = 0; i < numOfPoints; i++) {

		if (group->points[i].y < minY)
			minY = group->points[i].y;
		if (group->points[i].y > maxY)
			maxY = group->points[i].y;
		if (group->points[i].x < minX)
			minX = group->points[i].x;
		if (group->points[i].x > maxX)
			maxX = group->points[i].x;

	}

	// find corner points complicated
	for (int i = 0; i < numOfPoints; i++) {

		if (group->points[i].y == minY) {

			if (!firstMinYminX) {
				if (group->points[i].x < minYminX) {
					minYminX = group->points[i].x;
					cornerPoints[0] = group->points[i];
				}
			}
			else {
				firstMinYminX = false;
				minYminX = group->points[i].x;
				cornerPoints[0] = group->points[i];
			}

			if (group->points[i].x > minYmaxX) {
				minYmaxX = group->points[i].x;
				cornerPoints[1] = group->points[i];
			}
		}

		if (group->points[i].y == maxY) {

			if (!firstMaxYminX) {
				if (group->points[i].x < maxYminX) {
					maxYminX = group->points[i].x;
					cornerPoints[2] = group->points[i];
				}
			}
			else {
				firstMaxYminX = false;
				maxYminX = group->points[i].x;
				cornerPoints[2] = group->points[i];
			}

			if (group->points[i].x > maxYmaxX) {
				maxYmaxX = group->points[i].x;
				cornerPoints[3] = group->points[i];
			}
		}

		if (group->points[i].x == minX) {

			if (!firstMinXminY) {
				if (group->points[i].y < minXminY) {
					minXminY = group->points[i].y;
					cornerPoints[4] = group->points[i];
				}
			}
			else {
				firstMinXminY = false;
				minXminY = group->points[i].y;
				cornerPoints[4] = group->points[i];
			}

			if (group->points[i].y > minXmaxY) {
				minXmaxY = group->points[i].y;
				cornerPoints[5] = group->points[i];
			}
		}

		if (group->points[i].x == maxX) {

			if (!firstMaxXminY) {
				if (group->points[i].y < maxXminY) {
					maxXminY = group->points[i].y;
					cornerPoints[6] = group->points[i];
				}
			}
			else {
				firstMaxXminY = false;
				maxXminY = group->points[i].y;
				cornerPoints[6] = group->points[i];
			}

			if (group->points[i].y > maxXmaxY) {
				maxXmaxY = group->points[i].y;
				cornerPoints[7] = group->points[i];
			}
		}
	}

	group->klusterDiameter = 0;

	// first check: achieve virtual largest diameter o(1) max corner points = 8. make a virtual circle
	for (int i = 0; i < numOfCornerPoints; i++) {
		for (int j = 0; j < numOfCornerPoints; j++) {
			tempDiameter = calculateDistance(cornerPoints[i], cornerPoints[j]);
			if (tempDiameter > group->klusterDiameter) {
				group->klusterDiameter = tempDiameter;
				// get the virtual circle center
				circleCenter.x = (cornerPoints[i].x + cornerPoints[j].x) / 2;
				circleCenter.y = (cornerPoints[i].y + cornerPoints[j].y) / 2;
			}
		}
	}

	// second check: filter points
	for (int i = 0; i < numOfPoints; i++) {

		distanceFromCircleCenter = calculateDistance(circleCenter, group->points[i]);

		if (distanceFromCircleCenter >(group->klusterDiameter / 2)) {
				numOfSuspeciousPoints++;
		}
	}

	// allocate new suspecious points
	SuspeciousPoints = (Point*)malloc((numOfSuspeciousPoints+numOfCornerPoints) * sizeof(Point));

	// corner points are suspcious points
	for (int i = 0; i < numOfCornerPoints; i++)
		SuspeciousPoints[i] = cornerPoints[i];

	// assign filtered points
	for (int i = 0; i < numOfPoints; i++) {

		distanceFromCircleCenter = calculateDistance(circleCenter, group->points[i]);

		if (distanceFromCircleCenter > (group->klusterDiameter / 2)) {

				SuspeciousPoints[counterSuspeciousPoints] = group->points[i];
				counterSuspeciousPoints++;

		}
	}

	// achieve largest diameter
	tempGroup = (Group*)malloc(sizeof(Group));
	tempGroup->numOfPoints = (numOfSuspeciousPoints+numOfCornerPoints);
	tempGroup->points = SuspeciousPoints;
	findGroupDiameterWithCuda(tempGroup);

	// last check if suspicious group has bigger diameter
	if (tempGroup->klusterDiameter > group->klusterDiameter)
		group->klusterDiameter = tempGroup->klusterDiameter;

	//printf("klusterDiameter: %.3f \n", group->klusterDiameter);

	// free allocation
	free(SuspeciousPoints);
	free(tempGroup);
}

float kMeansAlgorithm(int numOfKlusters, float numOfPoints, float maxIterations, Point *points, Group *groups) {

	float qualityMeasure = 0;
	int numOfIterations = 0;
	int i = 0, j = 0;
	double start, end;

	// allocate memory
	Kluster *klusters = (Kluster*)malloc(numOfKlusters * sizeof(Kluster));

	// 2. initiate first k points
	for (j = 0; j < numOfKlusters; j++) {
		klusters[j].point = points[j];
	}

	// iterations of k-means algorithm with k number of cluster
	for (numOfIterations; numOfIterations < maxIterations; numOfIterations++) {

		//3. group points to given k cluster (calculate distances and group) and create k new number of float memory allocations for each group
			groupPointsToKlusters(klusters, points, groups, numOfKlusters, numOfPoints);

		//4. Recalculate the cluster centers – average of all points in the cluster
		calculateNewKlustersCenterOfMass(groups, numOfKlusters);

		//5. Check the termination condition , if new clusters have moved
		int count = 0;
		for (i = 0; i < numOfKlusters; i++) {
			if (klusters[i].point.x == groups[i].kluster.point.x &&
				klusters[i].point.y == groups[i].kluster.point.y)
				count++;
		}
		if (count == numOfKlusters)
			break;

		//6. assign new klusters from groups and repeat
		for (i = 0; i < numOfKlusters; i++)
			klusters[i] = groups[i].kluster;
	}

	//7. Evaluate the quality of the clusters
	// calculate distance from 2 farthest points
#pragma omp parallel for private(i)
	for (int i=0; i<numOfKlusters; i++)
		findGroupDiamaeterWithAlgorithm(&groups[i]);

	// caculate quality measure
	qualityMeasure = calculateQualityMeasure(groups, numOfKlusters);

	//free memory
	for (i = 0; i < numOfKlusters; i++) {
		free(groups[i].points);
	}
	free(klusters);

	return qualityMeasure;
}


