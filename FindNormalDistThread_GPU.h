/*
 * FindNormalDistBruteGPU.h
 *
 *  Created on: Jun 22, 2016
 *      Author: Brian Donohue
 *
 *  CUDA C version of GM_Normal.cpp function
 *
 *  FindNormalDistThreads
 ///  Copyright (C) 2007 by Bill Rogers, Research Imaging Center, UTHSCSA
///  rogers@uthscsa.edu
 *
 */

#ifndef FINDNORMALDISTTHREAD_GPU_H_
#define FINDNORMALDISTTHREAD_GPU_H_
#include "RicMesh.h"
#include "RicTexture.h"

typedef struct{
	vertex * vertices;
	triangle * polygons;
	vertex * normals;
}GPU_RicMesh;

int FindNormalDistThreads_GPU(RicMesh *inner_mesh, RicMesh *outer_mesh, RicMesh *closest_vects,
					RicTexture *thick, int nsub, float over, int nflip, float mind, float maxd);
void FindNormalDistThreads_Single_Thread(GPU_RicMesh  inner_mesh, GPU_RicMesh outer_mesh, GPU_RicMesh  output_mesh, float * texture_map,
		          int * inner_vlist, int * inner_plist, int * outer_plist,  int nflip,  float mind, float maxd, int n_vertices, int n_polys, int outer_n_polys);


#endif /* FINDNORMALDISTBRUTEGPU_H_ */
