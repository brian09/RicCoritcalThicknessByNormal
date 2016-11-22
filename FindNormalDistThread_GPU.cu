/*
 * FindNormalDistButeGPU.cpp
 * Based on FindNormalDistBute function in GM_Normal.cpp by Bill Rogers
 */
#include "FindNormalDistThread_GPU.h"
#include "RicMesh.h"
#include "RicTexture.h"
#include <cmath>
#include <stdio.h>
#include <float.h>
#ifndef ERRVAL
#define ERRVAL (float)9999
#endif

#define blockSize1 1024

#define blockSize2_x 32
#define blockSize2_y 8




#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{

   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

      if (abort) exit(code);
   }
}


__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int *lock){
  while (atomicCAS((int *)lock, 0, 1) != 0);
  }

__device__ void release_semaphore(volatile int *lock){
  *lock = 0;
  __threadfence();
  }


__device__ float fatomicMin(float *addr, float value){

        float old = *addr, assumed;

        if(old <= value) return old;

        do{

                assumed = old;

                old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

        }while(old!=assumed);

        return old;

}





/*
 * setNormLines
 * Calculate normal line from vertex to a normal and initializes
 * many arrays to their default values
 *
 * Params:
 * vertices - inner mesh vertex array
 * normals - inner mesh normal vertex array
 * p1 - point containing calculated vertex and normals
 * p0 - point containing vertex from vertices
 * inner_vlist - inner mesh vertex index list
 * keep_going - boolean that decides to keep that thread going in later kernel calls
 * outdist - array of distances to the nearest outer mesh polygon
 * indist - array of distances to the nearest inner mesh polygon
 * sfac - factor that multiplies the normal vertices
 * n_vertices - number of vertices
 */
__global__ void setNormLines(vertex * vertices, vertex * normals, Point * p1, Point * p0, int * inner_vlist, bool * keep_going, float * outdist,float * indist, float sfac, int n_vertices){

	int idx = threadIdx.x + blockIdx.x*blockDim.x;



	if(idx < n_vertices){

		if(vertices[inner_vlist[idx]].label == 1){

			keep_going[idx] = false;

		    Point p_0 = vertices[inner_vlist[idx]].pnt;
			p0[idx] = p_0;
		    Point n0 = normals[inner_vlist[idx]].pnt;

		    p1[idx].x = p_0.x + n0.x*sfac;
		    p1[idx].y = p_0.y + n0.y*sfac;
		    p1[idx].z = p_0.z + n0.z*sfac;

		    indist[idx] = ERRVAL;
                    outdist[idx] = ERRVAL;
		}else{
	            keep_going[idx] = true;
	            Point p_0 = vertices[inner_vlist[idx]].pnt;
		    p0[idx] = p_0;
	            Point n0 = normals[inner_vlist[idx]].pnt;

	            p1[idx].x = p_0.x + n0.x*sfac;
	            p1[idx].y = p_0.y + n0.y*sfac;
	            p1[idx].z = p_0.z + n0.z*sfac;

	           indist[idx] = ERRVAL;
	           outdist[idx] = ERRVAL;
		}
	}
}

typedef struct T_Points{

	Point *t0;
	Point *t1;
	Point *t2;


}T_Points;


/*
 * setTPoimts
 * Calculates tests distances from each polygons vertex to p0
 * in order to see if the thread should be used
 *
 * Params:
 * polygons- array of mesh triangles
 * vertices- array of mesh vertices
 * p0 - array points at a certain index
 * max - value equal to maxd*maxd
 * keep_going - keeps the xidx thread moving forward or stopping it
 * keep_going_tot - keeps the total index moving forward
 * plist - polygon index list
 * n_polys - number of polygons
 * n_vertices - number of vertices
 */
__global__ void setTPoints(triangle * polygons,vertex * vertices, Point * p0,
		float max ,bool * keep_going, bool * keep_going_tot, int * plist,
		int n_polys, int n_vertices){

	int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
	int yIdx = threadIdx.y + blockDim.y*blockIdx.y;
	int totIdx = xIdx + yIdx*n_vertices;
	if((xIdx < n_vertices) && (yIdx < n_polys)){
		keep_going_tot[totIdx] = true;
		if(keep_going[xIdx] == false){
			keep_going_tot[totIdx] = false;
			return;
		}
			Point t0 = vertices[polygons[plist[yIdx]].vidx[0]].pnt;
			Point t1 = vertices[polygons[plist[yIdx]].vidx[1]].pnt;
			Point t2 = vertices[polygons[plist[yIdx]].vidx[2]].pnt;


			Point p_0 = p0[xIdx];


			float d0 = (p_0.x-t0.x)*(p_0.x-t0.x) + (p_0.y-t0.y)*(p_0.y-t0.y) +
					(p_0.z-t0.z)*(p_0.z-t0.z);


			float d1 = (p_0.x-t1.x)*(p_0.x-t1.x) + (p_0.y-t1.y)*(p_0.y-t1.y) +
					(p_0.z-t1.z)*(p_0.z-t1.z);

			float d2 = (p_0.x-t2.x)*(p_0.x-t2.x) + (p_0.y-t2.y)*(p_0.y-t2.y) +
					(p_0.z-t2.z)*(p_0.z-t2.z);



			if(d0 < 0.1){
				keep_going_tot[totIdx] = false;
				return;
			}

			if(d1 < 0.1){
				keep_going_tot[totIdx] = false;
				return;
			}

			if(d2 < 0.1){
				keep_going_tot[totIdx] = false;
				return;
			}

			if(d0>max && d1>max && d2>max){
				keep_going_tot[totIdx] = false;
				return;

			}





	}

}



/* line_intersect_triangle
 * Taken from Ben Trumbore's code from the Journal of Graphics Tools (JGT)
 * This version Finds the point of intersection between and line segment and
 * a triangle rather than the intersection of a ray with the triangle.
 *
 * Params:
 * polygons - array of mesh triangles
 * vertices - array of mesh vertices
 * p0 - array of points from mesh
 * p1 - array of points of extended normal at corresponding p0 point
 * pout - point that intersects line segment and triangle
 * keep_going - array of booleans that determines if thread should keep calculating
 * plist - triangle index array
 * n_vertices - number of vertices
 * n_polys - number of triangles
 */
__global__ void line_intersect_triangle(triangle  * polygons, vertex * vertices, Point * p0, Point * p1, Point * pout,
		bool * keep_going, int * plist, int n_vertices, int n_polys)
{

	int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
	int yIdx = threadIdx.y + blockDim.y*blockIdx.y;
	int totIdx = xIdx + yIdx*n_vertices;

	if((xIdx<n_vertices) && (yIdx < n_polys)){




		Point t0 = vertices[polygons[plist[yIdx]].vidx[0]].pnt;
		Point t1 = vertices[polygons[plist[yIdx]].vidx[1]].pnt;
		Point t2 = vertices[polygons[plist[yIdx]].vidx[2]].pnt;





		Point p_0 = p0[xIdx];
		Point p_1 = p1[xIdx];


		Point d;
		d.x = p_1.x - p_0.x;
		d.y = p_1.y - p_0.y;
		d.z = p_1.z - p_0.z;

		Point e1;
		e1.x = t1.x - t0.x;
		e1.y = t1.y - t0.y;
		e1.z = t1.z - t0.z;

		Point e2;
		e2.x = t2.x - t0.x;
		e2.y = t2.y - t0.y;
		e2.z = t2.z - t0.z;

		Point h = d.CrossProduct(e2);

		float a = e1.Dot(h);

		if (a > -0.00001 && a < 0.00001){
			keep_going[totIdx] = false;
			return;

		}

		float f = 1/a;

		Point s;
		s.x = p_0.x - t0.x;
		s.y = p_0.y - t0.y;
		s.z = p_0.z - t0.z;
		float u = f * (s.Dot(h));

		if(u < 0.0 || u > 1.0){
			keep_going[totIdx] = false;
			return;

		}

		Point q = s.CrossProduct(e1);
		float v = f * d.Dot(q);

		if(v < 0.0 || (u + v)>1.0){
			keep_going[totIdx] = false;
			return;

		}

		float t = f * e2.Dot(q);

		if(t < 0.00001){
			keep_going[totIdx] = false;
			return;
		}

		Point l_pout = (p_0 + d*t);

		float dseg = (p_0.x-p_1.x)*(p_0.x-p_1.x) + (p_0.y-p_1.y)*(p_0.y-p_1.y) +
				(p_0.z-p_1.z)*(p_0.z-p_1.z);

		float dparts = (p_0.x-l_pout.x)*(p_0.x-l_pout.x) + (p_0.y-l_pout.y)*(p_0.y-l_pout.y) + (p_0.z-l_pout.z)*(p_0.z-l_pout.z)
+ (p_1.x-l_pout.x)*(p_1.x-l_pout.x) + (p_1.y-l_pout.y)*(p_1.y-l_pout.y) + (p_1.z-l_pout.z)*(p_1.z-l_pout.z);

		float w = dparts - dseg;



		if(w < 0.00001){
			pout[totIdx] = l_pout;
		}else{
			keep_going[totIdx] = false;
			pout[totIdx] = l_pout;
		}

	}

}

/*
 * calculate_inner_dist
 * Calculates inner distances from p0 to pin
 *
 * Params:
 * p0 - array of points from inner mesh
 * pin - point that corresponding p0's normal intersects
 * keepGoing - determiens whether current thread should continue
 * indist - array of inner distances
 * n_vertices - number of vertices
 * n_poly - number of triangles
 */
__global__ void calculate_inner_dist(Point * p0, Point * pin, bool * keepGoing,
		float * indist, int n_vertices , int n_poly){

	int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
	int yIdx = threadIdx.y + blockDim.y*blockIdx.y;
	int totIdx = xIdx + yIdx*n_vertices;
	__shared__ float indists[blockSize1];


		float d = ERRVAL;
		Point l_pin;
		Point l_p0;
		if(xIdx < n_vertices && yIdx < n_poly){
			if(keepGoing[totIdx] == true){
			    l_pin = pin[totIdx];
			    l_p0 = p0[xIdx];
			    d = sqrt((l_pin.x-l_p0.x)*(l_pin.x-l_p0.x) + (l_pin.y-l_p0.y)*(l_pin.y-l_p0.y) +
					(l_pin.z-l_p0.z)*(l_pin.z-l_p0.z));
			    if(d == 0.f)
			    	d=ERRVAL;

			}else{
				d = ERRVAL;
			}
		}else {
			d = ERRVAL;
		}


		indists[threadIdx.y] = d;
		__syncthreads();



	      for (unsigned int stride = blockDim.y/2; stride > 0; stride >>= 1) {


	            if((threadIdx.y <  stride) && ((yIdx + stride)< n_poly) && (xIdx < n_vertices)){
		    		if(indists[threadIdx.y] >  indists[threadIdx.y+stride]){
		    			indists[threadIdx.y] = indists[threadIdx.y+stride];
		    		}

	            }
	            __syncthreads();
	        }




	      if((threadIdx.y == 0) && (xIdx < n_vertices) && (yIdx < n_poly)){

	    		  fatomicMin(&indist[xIdx], indists[0]);

	      }




}


/*
 * calculate_outer_dist
 * Calculates outer distances from p0 to pin
 * and the array of minimum points
 *
 * Params:
 * p0 - array of points from inner mesh
 * pin - array of points that correspond to p0's normal intersections
 * minpnt - array of minimum points for outer polygon mesh
 * keepGoing - determiens whether current thread should continue
 * noutfound - number of valid distances found
 * outdist - array outer mesh distances
 * indist - array of inner distances
 * maxd - maximum valid distance to be used
 * n_vertices - number of vertices
 * n_poly - number of triangles
 */
__global__ void calculate_outer_dist( Point * p0, Point * pin, Point * minpnt, bool * keepGoing,  int * noutfound, float * outdist,
		float * indist, float maxd, int n_vertices , int n_poly){

	int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
	int yIdx = threadIdx.y + blockDim.y*blockIdx.y;
	int totIdx = xIdx + yIdx*n_vertices;
	__shared__ float outdists[blockSize1];
	__shared__ int pntfound[blockSize1];


		Point l_pin;
		Point l_p0;
		float d =ERRVAL;
		if(xIdx<n_vertices && yIdx<n_poly){
			if(keepGoing[totIdx] == true){

				l_pin = pin[totIdx];
			    l_p0 = p0[xIdx];
			    d = sqrt((l_pin.x-l_p0.x)*(l_pin.x-l_p0.x) + (l_pin.y-l_p0.y)*(l_pin.y-l_p0.y) +
					(l_pin.z-l_p0.z)*(l_pin.z-l_p0.z));
			    pntfound[threadIdx.y] = 1;
			}else{
				d= ERRVAL;
				pntfound[threadIdx.y] = 0;
			}
		}else {
			pntfound[threadIdx.y] = 0;
			d = ERRVAL;
		}




		outdists[threadIdx.y] = d;
		__syncthreads();



	    for (unsigned int stride = blockDim.y/2; stride > 0; stride >>= 1) {
	    	if((threadIdx.y <  stride) && ((yIdx + stride)< n_poly)  && (xIdx < n_vertices)){
	    		pntfound[threadIdx.y] += pntfound[threadIdx.y + stride];
	    		if(outdists[threadIdx.y] >  outdists[threadIdx.y+stride]){
	    			outdists[threadIdx.y] = outdists[threadIdx.y+stride];
	    		}
	    	}
	            __syncthreads();
	    }


	      if(xIdx<n_vertices && threadIdx.y == 0 && yIdx < n_poly){
	    	  if(keepGoing[totIdx] == true){
	    		  atomicAdd(&noutfound[xIdx], 1);
	    	  }
	      }

	      if(threadIdx.y != 0)
		return;


	      float l_outdist = outdists[0];
	      __syncthreads();
	      if((threadIdx.y == 0) && (xIdx < n_vertices) && (yIdx < n_poly))
		acquire_semaphore(&sem);
	      __syncthreads();

	      if((threadIdx.y == 0) && (xIdx < n_vertices) && (yIdx < n_poly)){
		  if(l_outdist <indist[xIdx]  && l_outdist <maxd){
		      if(l_outdist < outdist[xIdx]){
			  outdist[xIdx] = l_outdist;
	    	          minpnt[xIdx].x =l_pin.x;
	                  minpnt[xIdx].y =l_pin.y;
	    		  minpnt[xIdx].z = l_pin.z;
		      }
		  }
	      }
	      __syncthreads();
	      if ((threadIdx.y == 0) && (xIdx < n_vertices) && (yIdx < n_poly))
		release_semaphore(&sem);
	      __syncthreads();


	//}

}


/*
 * check_min_dist
 * Checks minimum outer distances and adjusts them
 * accordingly then sets it's corresponding texture node
 * equal to it.
 *
 * Params:
 * outdist - array of outer distances from outer mesh
 * nodes - array of texture nodes
 * noutfound - array of minimum points from outer polygon mesh
 * node_in - determiens whether current thread should continue
 * mind - minumum valid distance to be used
 * maxd - maximum valid distance to be used
 * inner_vlist - inner vertex index list
 * n_vertices - number of vertices
 */
__global__ void check_min_dist(float * outdist, float * nodes, int * noutfound, bool * node_in,
		                  float mind, float maxd,  int * inner_vlist, int n_vertices){
	int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
	if(xIdx < n_vertices){
		if(node_in[xIdx] == true){


		float l_outdist = outdist[xIdx];

		if(l_outdist < mind)
			l_outdist = mind;

		if((l_outdist == ERRVAL) && (noutfound[xIdx]>0))
			l_outdist = maxd;

		nodes[inner_vlist[xIdx]] = l_outdist;
		}
	}
}


/*
 * assign_vector_for_points
 * Assigns points to an output mesh
 *
 * Params:
 * in_vertices - array of inner mesh vertices
 * in_nomrals - array of  inner mesh normals
 * out_vertices - array of output mesh vertices
 * out_normals - array of output mesh normals
 * out_polys - array of output mesh polygons
 * minpnt - array of minimum points that have been calculated=
 * mind - minimum valid distance to be used
 * maxd - maximum valid distance to be used
 * inner_vlist - inner vertex index list
 * n_vertices - number of vertices
 */
__global__ void assign_vector_for_points(vertex * in_vertices, vertex * in_normals, vertex * out_vertices, vertex * out_normals,
		                         triangle * out_polys, Point * minpnt, float * outdist,  bool * keep_going, float maxd, float mind,
		                         int * inner_vlist, int n_vertices){
	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;


	if(xIdx < n_vertices){
	    if(keep_going[xIdx] == true){
		float l_outdist = outdist[xIdx];

		if(l_outdist!=ERRVAL && l_outdist>mind && l_outdist<maxd){
			Point cvert(minpnt[xIdx].x, minpnt[xIdx].y, minpnt[xIdx].z);
			int n = inner_vlist[xIdx];
			out_vertices[2*n].pnt = in_vertices[n].pnt;
			Point l_normal = in_normals[n].pnt;
			out_normals[2*n].pnt = l_normal;
			out_normals[2*n + 1].pnt = l_normal;
			out_vertices[2*n + 1].pnt = cvert;
			out_polys[n].assign(2*n, 2*n+1, 0);
		}
	    }
	}
}


/*
 * initialize_dist_and_norm_lines
 * Initializes distances that are calculated later and
 * points p0 and p1.  p0 being a point from vertices and p1 being
 * p0 - normal*sfac.
 *
 * Params:
 * vertices - array of inner mesh vertices
 * normals - array of inner mesh normals
 * p1 - array of vertex point with extended normal
 * p0 - array of vertex point
 * keep_going - boolean values that determine if x thread should keep going
 * indist - inner distance values calculated later but initialized here
 * outdist- outer distance values calculated later but initialized her
 * sfac - scale factor for normal extended at p0
 * vlist - inner vertex list
 * n_vertices - number of vertices to be used
 */
void initialize_dist_and_norm_lines(vertex * vertices, vertex * normals, Point * p1, Point * p0,
		                           bool * keep_going, float * indist, float * outdist, float sfac,
		                           int * vlist, int n_vertices){

	dim3 blockSize(blockSize1, 1, 1);
	int gridSizeX = ceil(float(n_vertices)/float(blockSize1));
	dim3 gridSize(gridSizeX, 1, 1);



	setNormLines<<<gridSize, blockSize>>>(vertices, normals, p1, p0, vlist,keep_going, outdist, indist ,sfac,  n_vertices);
	gpuErrchk( cudaPeekAtLastError() );


}

/*
 * inner_mesh_calculations
 * Calculates indist array
 *
 * Params:
 * polys - array of array of inner mesh triangles
 * vertices - array of inner mesh vertices
 * p0 - array of points from vertices
 * p1 - array of points that are the array of p0 points extended from the normal n0
 * indist - distance from p0 to the closest vertex on the triangle
 * keep_going_x - boolean for x values to keep going
 * maxd - maximum allowed distance to be used
 * plist - polygon index list
 * n_vertices - number of vertices
 * n_polys - number of polygons
 */
 void inner_mesh_calculations(triangle * polys, vertex * vertices, Point * p0, Point * p1, float * indist,
		 bool * keep_going_x, float maxd,int * plist,  int n_vertices, int n_polys){
   bool * keep_going_tot;
   Point * pout;
   dim3 blockSize(blockSize2_x, blockSize2_y, 1);
   int gridSizeX = ceil(float(n_vertices)/float(blockSize2_x));
   int gridSizeY = ceil(float(n_polys)/float(blockSize2_y));
   dim3 gridSize(gridSizeX, gridSizeY, 1);

   gpuErrchk(cudaMalloc((void**)&keep_going_tot, sizeof(bool)*n_vertices*n_polys ));
   gpuErrchk(cudaMemset(keep_going_tot, 0, sizeof(bool)*n_vertices*n_polys ));
   float maxdsqu = maxd*maxd;


    setTPoints<<<gridSize, blockSize>>>(polys, vertices, p0, maxdsqu, keep_going_x, keep_going_tot, plist, n_polys, n_vertices);
    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk(cudaMalloc((void**)&pout, sizeof(Point)*n_vertices*n_polys ));


   line_intersect_triangle<<<gridSize, blockSize>>>(polys, vertices, p0, p1,  pout, keep_going_tot, plist,  n_vertices,  n_polys);
    gpuErrchk( cudaPeekAtLastError() );

    dim3 blockSize2(1, blockSize1, 1);
    int gridSizeX2 = n_vertices;
    int gridSizeY2 = ceil(float(n_polys)/float(blockSize1));
    dim3 gridSize2(gridSizeX2, gridSizeY2, 1);

    calculate_inner_dist<<<gridSize2, blockSize2>>>( p0, pout, keep_going_tot, indist, n_vertices , n_polys);

    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk(cudaFree(pout) );
    gpuErrchk(cudaFree(keep_going_tot) );

}

 /*
  * outer_mesh_calculations
  * Calculates outdist array and minpnt array
  *
  * Params:
  * polys - array of array of outer mesh triangles
  * vertices - array of outer mesh vertices
  * p0 - array of points from vertices
  * p1 - array of points that are the array of p0 points extended from the normal n0
  * minpnt - point closest to p0 normal
  * noutfound - number of outer mesh points found
  * outdist - distance from p0 to the closest vertex on the triangle for the outer mesh
  * indist - distance from p0 to the closest vertex on the triangle for the inner mesh
  * maxd - maximum allowed distance to be used
  * keep_going_x - boolean for x values to keep going
  * plist - polygon index list
  * n_vertices - number of vertices
  * n_polys - number of polygons
  */
void outer_mesh_calculations(triangle * polygons, vertex * vertices, Point * p0, Point * p1, Point * minpnt,
		int * noutfound, float * outdist, float * indist, float maxd, bool * keep_going_x,
		int * plist, int n_vertices, int n_polys){
  bool * keep_going_tot;
  Point * pin;
  dim3 blockSize(blockSize2_x, blockSize2_y, 1);
  int gridSizeX = ceil(float(n_vertices)/float(blockSize2_x));
  int gridSizeY = ceil(float(n_polys)/float(blockSize2_y));
  dim3 gridSize(gridSizeX, gridSizeY, 1);
  gpuErrchk(cudaMalloc((void**)&keep_going_tot, sizeof(bool)*n_vertices*n_polys ));
  gpuErrchk(cudaMemset(keep_going_tot, 0, sizeof(bool)*n_vertices*n_polys ));
  float maxsqu = maxd*maxd;
  setTPoints<<<gridSize, blockSize>>>(polygons, vertices, p0,
			maxsqu, keep_going_x,  keep_going_tot, plist, n_polys,  n_vertices);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk(cudaMalloc((void**)&pin, sizeof(Point)*n_vertices*n_polys ));

  line_intersect_triangle<<<gridSize, blockSize>>>(polygons,vertices, p0, p1, pin, keep_going_tot, plist, n_vertices,  n_polys);
  gpuErrchk( cudaPeekAtLastError() );
  dim3 blockSize2(1, blockSize1, 1);
  int gridSizeX2 = n_vertices;
  int gridSizeY2 = ceil(float(n_polys)/float(blockSize1));
  dim3 gridSize2(gridSizeX2, gridSizeY2, 1);

  calculate_outer_dist<<<gridSize2, blockSize2>>>(p0,  pin,  minpnt,  keep_going_tot,   noutfound,
			outdist, indist, maxd,  n_vertices ,  n_polys);
  gpuErrchk( cudaPeekAtLastError() );

  gpuErrchk(cudaFree(pin) );
  gpuErrchk(cudaFree(keep_going_tot) );

}

/*
 * calculate_dist_mesh_values
 * Calculates outdist which is use for the texture map and
 * assigns the minpnts to the output mesh
 *
 * Params:
 * out_polys - output mesh polygons
 * in_vertices - inner mesh vertices
 * out_vertices - output mesh vertices
 * in_normals - inner mesh normal vertices
 * out_normals - output mesh normal vertices
 * minpnt - minimum point array
 * nodes - texture map output nodes
 * outdist - outer mesh vertex distances
 * noutfound - number of vertices found
 * mind - minimum distance accepted
 * maxd - maximum distance accepted
 * node_in - boolean nodes in
 * inner_vlist - inner mesh vertex index array
 * n_vertices - number of vertices
 * n_polys - number of polygons
 */
void calculate_dist_mesh_values(triangle * out_polys, vertex * in_vertices, vertex * out_vertices,  vertex * in_normals,
		vertex * out_normals, Point * minpnt, float * nodes, float * outdist, int * noutfound, float mind,
		float maxd, bool * node_in, int * inner_vlist,  int n_vertices, int n_polys){

	dim3 blockSize(blockSize1, 1, 1);
	dim3 gridSize(ceil(float(n_vertices)/float(blockSize1)), 1, 1);

	check_min_dist<<<gridSize, blockSize>>>(outdist, nodes, noutfound, node_in, mind,  maxd,inner_vlist,   n_vertices);
	gpuErrchk( cudaPeekAtLastError() );


	assign_vector_for_points<<<gridSize, blockSize>>>(in_vertices, in_normals, out_vertices,  out_normals,  out_polys,
			        minpnt,  outdist,   node_in,  maxd, mind, inner_vlist, n_vertices);
	gpuErrchk( cudaPeekAtLastError() );

}


void FindNormalDistThreads_Single_Thread(GPU_RicMesh  inner_mesh, GPU_RicMesh outer_mesh, GPU_RicMesh  output_mesh, float * texture_map,
		          int * inner_vlist, int * inner_plist, int * outer_plist,  int nflip,  float mind, float maxd, int n_vertices, int n_polys, int outer_n_polys){





  	float sfac = nflip*maxd;

	Point * minpnt;
	int  * noutfound;
	float * indist, * outdist;
	Point * p0, * p1;

	bool * keep_going;



	gpuErrchk(cudaMalloc((void**)&p0, sizeof(Point)*n_vertices ));
	gpuErrchk(cudaMalloc((void**)&p1, sizeof(Point)*n_vertices ));

	gpuErrchk(cudaMalloc((void**)&keep_going, sizeof(bool)*n_vertices ));

	gpuErrchk(cudaMalloc((void**)&indist, sizeof(float)*n_vertices ));







	gpuErrchk(cudaMalloc((void**)&outdist, sizeof(float)*n_vertices ));


	gpuErrchk(cudaMemset(keep_going, 0, sizeof(bool)*n_vertices));




	initialize_dist_and_norm_lines(inner_mesh.vertices, inner_mesh.normals,  p1,  p0,  keep_going, indist, outdist, sfac, inner_vlist,  n_vertices);














	inner_mesh_calculations(inner_mesh.polygons, inner_mesh.vertices,  p0, p1, indist,
			                            keep_going,  maxd,inner_plist, n_vertices,  n_polys);


	gpuErrchk(cudaMalloc((void**)&noutfound, sizeof(int)*n_vertices));

	gpuErrchk(cudaMalloc((void**)&minpnt, sizeof(Point)*n_vertices ));
	gpuErrchk(cudaMemset(noutfound, 0, sizeof(int)*n_vertices));

	outer_mesh_calculations(outer_mesh.polygons, outer_mesh.vertices, p0,  p1,  minpnt, noutfound,
			 outdist,  indist,  maxd,  keep_going, outer_plist, n_vertices, outer_n_polys);

	gpuErrchk(cudaFree(p0));
	gpuErrchk(cudaFree(p1));

	gpuErrchk(cudaFree(indist));


	calculate_dist_mesh_values(output_mesh.polygons, inner_mesh.vertices,  output_mesh.vertices,  inner_mesh.normals,
			output_mesh.normals, minpnt, texture_map, outdist,  noutfound, mind, maxd,  keep_going,
            inner_vlist, n_vertices,  n_polys);




	gpuErrchk(cudaFree(minpnt));
	gpuErrchk(cudaFree(outdist));
	gpuErrchk(cudaFree(noutfound));
	gpuErrchk(cudaFree(keep_going));








}





int FindNormalDistThreads_GPU(RicMesh *inner_mesh, RicMesh *outer_mesh, RicMesh *closest_vects,
					RicTexture *thick, int nsub, float over, int nflip, float mind, float maxd)
{


	int i,j,k,l,m;

        int nDevices;

	    cudaGetDeviceCount(&nDevices);

	    if(nDevices == 0){
		cout << "No CUDA capable Device Detected\n";
		exit(1);
	    }
	    cudaDeviceProp prop;

	    cudaGetDeviceProperties(&prop, 0);

	    if(prop.major < 2){
		cout << "CUDA device architecture detected is less than 2.0X\n";
		cout << prop.name << "\n";
		cout << "Device Architecture: " << prop.major << "." << prop.minor << "X\n";
		exit(1);
	    }




	// check to see that the inner mesh and the texture are the same size
	if ( inner_mesh->v_size != thick->size )
	{
		cout << "FindNormalDistThreads - error inner mesh and texture not same size"<<endl;
		return 0;
	}

	// array pointers for vertex and polygon lists for subdividing the meshes
	int **inv_mat;	// inner mesh vertices
	int **inp_mat;	// inner mesh polygons
	int **outp_mat;	// outer mesh polygons

	// number of threads
	int nthreads = nsub*nsub*nsub;

	// allocate memory for the arrays for vertices and polygons
	matrix(&inv_mat,nthreads,inner_mesh->v_size); // inner vertices
	matrix(&inp_mat,nthreads,inner_mesh->p_size); // inner polygons
	matrix(&outp_mat,nthreads,outer_mesh->p_size); // outer polygons

	// allocate thread pointers


	// allocate array of structures passing data to threads


	// scale factor for normal line end point - includes normal direction
	float sfac = nflip*maxd;

	// make sure that limits have been calculated for the gm mesh
	inner_mesh->calc_limits();

	// find the step size for each axis for an iteration
	float xinc,yinc,zinc;
	xinc = (inner_mesh->xmax-inner_mesh->xmin)/(float)nsub;
	yinc = (inner_mesh->ymax-inner_mesh->ymin)/(float)nsub;
	zinc = (inner_mesh->zmax-inner_mesh->zmin)/(float)nsub;

	// initialize the thickness values to ERRVAL
	for ( i=0 ; i<thick->size ; ++i ) thick->nodes[i]=ERRVAL;

	// the number of actual searches will be the cube of the subdivision number
	float xstart,ystart,zstart;	// starting values for vertex search
	float xend,yend,zend;		// ending value for vertex search
	float xstart2,ystart2,zstart2;	// starting values for vertex search
	float xend2,yend2,zend2;		// ending value for vertex search
	int n_in_verts = 0;	// number of inner mesh vertices found in a subdivision
	int n_in_polys = 0;	// number of inner mesh polygons found in a subdivision
	int n_out_polys = 0;// number of outer mesh polygons found in a subdivision
	int cur_thread;


	GPU_RicMesh  d_inner_mesh;
	GPU_RicMesh  d_closest_vects;
	GPU_RicMesh  d_outer_mesh;

	gpuErrchk(cudaMalloc((void**)&d_inner_mesh, sizeof(GPU_RicMesh)));
	gpuErrchk(cudaMalloc((void**)&d_outer_mesh, sizeof(GPU_RicMesh)));
	gpuErrchk(cudaMalloc((void**)&d_closest_vects, sizeof(GPU_RicMesh)));
	gpuErrchk(cudaMalloc((void**)&d_inner_mesh.vertices, inner_mesh->v_size*sizeof(vertex)));

	gpuErrchk(cudaMemcpy(d_inner_mesh.vertices, inner_mesh->vertices, inner_mesh->v_size*sizeof(vertex), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc((void**)&d_inner_mesh.normals, inner_mesh->n_size*sizeof(vertex)));
	gpuErrchk(cudaMemcpy(d_inner_mesh.normals, inner_mesh->normals, inner_mesh->n_size*sizeof(vertex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&d_inner_mesh.polygons, inner_mesh->p_size*sizeof(triangle)));
	gpuErrchk(cudaMemcpy(d_inner_mesh.polygons, inner_mesh->polygons, inner_mesh->p_size*sizeof(triangle), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc((void**)&d_outer_mesh.vertices, outer_mesh->v_size*sizeof(vertex)));


	gpuErrchk(cudaMemcpy(d_outer_mesh.vertices, outer_mesh->vertices, outer_mesh->v_size*sizeof(vertex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&d_outer_mesh.polygons, outer_mesh->p_size*sizeof(triangle)));
	gpuErrchk(cudaMemcpy(d_outer_mesh.polygons, outer_mesh->polygons, outer_mesh->p_size*sizeof(triangle), cudaMemcpyHostToDevice));



	gpuErrchk(cudaMalloc((void**)&d_closest_vects.vertices, closest_vects->v_size*sizeof(vertex)));
	gpuErrchk(cudaMemcpy(d_closest_vects.vertices, closest_vects->vertices, closest_vects->v_size*sizeof(vertex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&d_closest_vects.normals, closest_vects->n_size*sizeof(vertex)));
	gpuErrchk(cudaMemcpy(d_closest_vects.normals, closest_vects->normals, closest_vects->n_size*sizeof(vertex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&d_closest_vects.polygons, closest_vects->p_size*sizeof(triangle)));
        gpuErrchk(cudaMemcpy(d_closest_vects.polygons, closest_vects->polygons, closest_vects->p_size*sizeof(vertex), cudaMemcpyHostToDevice));


	float * d_thick;
	gpuErrchk(cudaMalloc((void**)&d_thick, thick->size*sizeof(float)));

	gpuErrchk(cudaMemcpy(d_thick,thick->nodes,sizeof(float)*thick->size, cudaMemcpyHostToDevice));

	////// repeat for each subdivision //////
	for ( i=0 ; i<nsub ; ++i )
	{
		xstart = inner_mesh->xmin + i*xinc;
		xend = xstart+xinc;

		for ( j=0 ; j<nsub ; ++j )
		{
			ystart = inner_mesh->ymin + j*yinc;
			yend = ystart+yinc;

			for ( k=0 ; k<nsub ; ++k )
			{
				///////////////// preprocess for current subdivision ///////////
				// find all the vertices and polygons in these bounds
				cur_thread = (i*nsub*nsub)+j*nsub+k;

				n_in_verts = n_in_polys = n_out_polys = 0;

				zstart = inner_mesh->zmin + k*zinc;
				zend = zstart + zinc;

				// find the appropriate gm vertices in this box
				for ( l=0 ; l<inner_mesh->v_size ; ++l )
				{
					if ( inner_mesh->vertices[l].pnt.x >= xstart && inner_mesh->vertices[l].pnt.x <= xend
						&&	inner_mesh->vertices[l].pnt.y >= ystart && inner_mesh->vertices[l].pnt.y <= yend
						&&	inner_mesh->vertices[l].pnt.z >= zstart && inner_mesh->vertices[l].pnt.z <= zend )
					{
						inv_mat[cur_thread][n_in_verts++] = l;
					}
				}

				// allow for overlap
				xstart2 = xstart-over;
				ystart2 = ystart-over;
				zstart2 = zstart-over;
				xend2 = xend+over;
				yend2 = yend+over;
				zend2 = zend+over;

				// look for inner triangles that fit in this box
				for ( l=0 ; l<inner_mesh->p_size ; ++l )
				{
					// check each triangle vertex
					for ( m=0 ; m<3 ; ++m )
					{
						int vidx = inner_mesh->polygons[l].vidx[m];
						if ( inner_mesh->vertices[vidx].pnt.x >= xstart2
							&& inner_mesh->vertices[vidx].pnt.x <= xend2
							&&	inner_mesh->vertices[vidx].pnt.y >= ystart2
							&& inner_mesh->vertices[vidx].pnt.y <= yend2
							&&	inner_mesh->vertices[vidx].pnt.z >= zstart2
							&& inner_mesh->vertices[vidx].pnt.z <= zend )
						{
							inp_mat[cur_thread][n_in_polys++] = l;
							break; // no need to check the others
						}
					}
				}

								// look for outer triangles that fit in this box
				for ( l=0 ; l<outer_mesh->p_size ; ++l )
				{
					// check each triangle vertex
					for ( m=0 ; m<3 ; ++m )
					{
						int vidx = outer_mesh->polygons[l].vidx[m];
						if ( outer_mesh->vertices[vidx].pnt.x >= xstart2
											   && outer_mesh->vertices[vidx].pnt.x <= xend2
											   &&	outer_mesh->vertices[vidx].pnt.y >= ystart2
											   && outer_mesh->vertices[vidx].pnt.y <= yend2
											   &&	outer_mesh->vertices[vidx].pnt.z >= zstart2
											   && outer_mesh->vertices[vidx].pnt.z <= zend2 )
						{
							outp_mat[cur_thread][n_out_polys++] = l;
							break; // no need to check the others
						}
					}
				}

				int * d_inv_mat;
				gpuErrchk(cudaMalloc((void**)&d_inv_mat, n_in_verts*sizeof(int)));
				int * d_inp_mat;
				gpuErrchk(cudaMalloc((void**)&d_inp_mat, n_in_polys*sizeof(int)));
				int * d_outp_mat;
				gpuErrchk(cudaMalloc((void**)&d_outp_mat, n_out_polys*sizeof(int)));

				gpuErrchk(cudaMemcpy(d_inv_mat, inv_mat[cur_thread], sizeof(int)*n_in_verts, cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(d_inp_mat, inp_mat[cur_thread], sizeof(int)*n_in_polys, cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(d_outp_mat, outp_mat[cur_thread], sizeof(int)*n_out_polys, cudaMemcpyHostToDevice));


				FindNormalDistThreads_Single_Thread(d_inner_mesh, d_outer_mesh, d_closest_vects, d_thick, d_inv_mat,
						d_inp_mat, d_outp_mat, nflip, mind, maxd, n_in_verts, n_in_polys, n_out_polys);




				gpuErrchk(cudaFree(d_inv_mat));
				gpuErrchk(cudaFree(d_inp_mat));
				gpuErrchk(cudaFree(d_outp_mat));


			} // z search
		} // y search
	} // x search

	// now wait for all the threads to finish
	gpuErrchk(cudaFree(d_inner_mesh.normals));
	gpuErrchk(cudaFree(d_inner_mesh.vertices));
	gpuErrchk(cudaFree(d_inner_mesh.polygons));



	gpuErrchk(cudaFree(d_outer_mesh.vertices));
	gpuErrchk(cudaFree(d_outer_mesh.polygons));

	gpuErrchk(cudaMemcpy(closest_vects->polygons, d_closest_vects.polygons, sizeof(triangle)*closest_vects->p_size, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(closest_vects->vertices, d_closest_vects.vertices, sizeof(vertex)*closest_vects->v_size, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(closest_vects->normals, d_closest_vects.normals, sizeof(vertex)*closest_vects->v_size, cudaMemcpyDeviceToHost));
	//std::cout << closest_vects->normals[0].pnt.x << "\n";
	gpuErrchk(cudaMemcpy(thick->nodes, d_thick, sizeof(float)*thick->size, cudaMemcpyDeviceToHost));
	//std::cout << thick->nodes[0] << "\n";
	gpuErrchk(cudaFree(d_closest_vects.polygons));
	gpuErrchk(cudaFree(d_closest_vects.vertices));
	gpuErrchk(cudaFree(d_closest_vects.normals));
	gpuErrchk(cudaFree(d_thick));


	// clean up memory allocation
	free_matrix(inv_mat);
	free_matrix(inp_mat);
	free_matrix(outp_mat);


	return 1;

}




