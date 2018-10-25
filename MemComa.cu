/*
1. Memory Copy Cost   One-Step
2. Straggler: Ring-based
**/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>
using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void HandleError( cudaError_t err, const char *file, int line )
{
	if (err != cudaSuccess)
	{
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit(-1);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
struct Block
{
	int block_id;
	int data_age;
	int sta_idx;
	int height; //height
	int ele_num;
	bool isP;
	double* eles;
	Block()
	{

	}
	Block operator=(Block& bitem)
	{
		block_id = bitem.block_id;
		data_age = bitem.data_age;
		height = bitem.height;
		eles = bitem.eles;
		ele_num = bitem.ele_num;
		sta_idx = bitem.sta_idx;
		return *this;
	}
	void printBlock()
	{

		printf("block_id  %d\n", block_id);
		printf("data_age  %d\n", data_age);
		printf("ele_num  %d\n", ele_num);
		for (int i = 0; i < ele_num; i++)
		{
			printf("%lf\t", eles[i]);
		}
		printf("\n");

	}
};
struct RatingEntry{
	int pidx;
	int qidx;
	double rate;
	RatingEntry(){

	}
	RatingEntry(int p, int q, double r){
		pidx = p;
		qidx = q;
		rate = r;
	}
	RatingEntry operator=(RatingEntry& ritem){
		pidx = ritem.pidx;
		qidx = ritem.qidx;
		rate = ritem.rate;
		return *this;
	}
};

//Yahoo!Music
#define FILE_NAME "./trainDS/"
#define TEST_NAME "./testDS"
#define N 1000990
#define M 624961
#define K 100 //主题个数
double yita = 0.001;
double theta = 0.05;


//#define SM_NUM 8
//#define MM_NUM 4

#define SM_NUM 4
#define MM_NUM 1
#define BK_NUM (SM_NUM*MM_NUM)
#define TD_NUM 1
#define BT_NUM (BK_NUM*TD_NUM)
#define RB_NUM (BT_NUM*BT_NUM)
#define ITER_CAP 500
Block PBlocks[BK_NUM], QBlocks[BK_NUM];
int random_seq[TD_NUM * ITER_CAP];
Block *dev_PBlocks, *dev_QBlocks;
int* dev_seq;
int* dev_flag;
double* dev_PData[BK_NUM], *dev_QData[BK_NUM];
double *dev_p_cache[BT_NUM], *dev_q_cache[BT_NUM];
std::map<long, double> EntryM;
vector<RatingEntry> Rblocks[BK_NUM*TD_NUM][BK_NUM*TD_NUM];
RatingEntry* dev_rate_entries[RB_NUM];
double entry_num[RB_NUM];
int* dev_entry_num;
int p_height, q_height, t_p_height, t_q_height;

void readTrainData();
void initParas();
void allocCudaMem();
void partitionP(int portion_num, int line_num,  Block * block_arr);
void freeCudaMem();

__global__ void MFkernel(Block* dev_PBlocks, Block* dev_QBlocks, double* dev_PData[], double* dev_QData[], double *dev_p_cache[],double *dev_q_cache[], int* dev_seq, int*dev_entry_num, RatingEntry* dev_rate_entries[], int p_height, int q_height, int* dev_flag, int epoch, double yita, double theta)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int iden = bid*TD_NUM+tid;
	printf("Test %d  %d  %d %d\n", tid, bid, i, iden );
	int iter = 0;

	Block* pblock = &(dev_PBlocks[bid]);
	Block* qblock = &(dev_QBlocks[BK_NUM-bid-1]);
	pblock->eles = dev_PData[bid];
	qblock->eles = dev_QData[BK_NUM-bid-1];
	//int row_unit = (pblock->ele_num)/TD_NUM;
	//int col_unit = (qblock->ele_num)/TD_NUM;
	int row, col, row_b_id, col_b_id, row_base, col_base, mini_b_idx, td_ele_num;
	int ele_idx, ele_p, ele_q, dimk;
	double ele_ra, error, sum_ra;
	RatingEntry* td_rate_entries;
	
	// one epoch
	row_b_id = bid;
	col_b_id = (BK_NUM - 1-row_b_id+epoch)%BK_NUM;

	row_base = row_b_id * BT_NUM;
	col_base = col_b_id *BT_NUM;

	for (iter = 0; iter < ITER_CAP; iter++)
	{
		//SGD
	 	row = row_base + tid ;
		col = col_base + dev_seq[iter*TD_NUM+tid];
		mini_b_idx = row*BT_NUM+col;
		td_rate_entries = dev_rate_entries[mini_b_idx];
		td_ele_num = dev_entry_num[mini_b_idx];

		for(ele_idx = 0; ele_idx<td_ele_num; ele_idx++){
			ele_p = (td_rate_entries[ele_idx].pidx)%p_height;
			ele_q = (td_rate_entries[ele_idx].qidx)%q_height;
			ele_ra = td_rate_entries[ele_idx].rate;
			sum_ra = 0;
			for(dimk = 0; dimk<K; dimk++){
				sum_ra+= dev_PData[row_b_id][ele_p*K+dimk]*dev_QData[col_b_id][ele_q*K+dimk];
				dev_p_cache[iden][dimk] = dev_PData[row_b_id][ele_p*K+dimk];
				dev_q_cache[iden][dimk] = dev_QData[col_b_id][ele_q*K+dimk];
			}
			error = ele_ra -sum_ra;
			//SGD
			for(dimk = 0; dimk < K; dimk++){
				dev_PData[row_b_id][ele_p*K+dimk] += yita * (error * dev_q_cache[iden][dimk] - theta * dev_p_cache[iden][dimk]);
				dev_QData[col_b_id][ele_q*K+dimk] += yita * (error * dev_p_cache[iden][dimk] - theta * dev_q_cache[iden][dimk]);
			}
		}
		//Sync
		__syncthreads();

	}

}

__global__ void helloFromGPU(void)
{
	printf("Hello from GPU\n");
}

void readTrainData()
{
	char fn[100];
	int i, j;
	for ( i = 0; i < 64; i++)
	{
		for (j = 0; j < 64; j++)
		{
			//iidx = i / (64 / BK_NUM);
			//jidx = j / (64 / BK_NUM);
			sprintf(fn, "%s%d-%d", FILE_NAME, i, j);
			ifstream ifs(fn);
			if (!ifs.is_open())
			{
				printf("Open fail %s\n", fn);
				exit(-1);
			}
			long hash_idx = -1;
			double ra = 0;
			while (!ifs.eof())
			{
				ifs >> hash_idx >> ra;
				if (hash_idx >= 0)
				{
					ra = ra / 100.0;
					//Rblocks[iidx][jidx].insert(pair<long, double>(hash_idx, ra));
					EntryM.insert(pair<long, double>(hash_idx, ra));

				}
			}

		}
		printf("row %d fini\n", i);
	}
	p_height = (N+BK_NUM-1) / BK_NUM;
	q_height = (M+BK_NUM-1)/BK_NUM;
	t_p_height = (p_height+TD_NUM-1)/TD_NUM;
	t_q_height = (q_height+TD_NUM-1)/TD_NUM;

	map<long, double>::iterator it; 
	for(it=EntryM.begin(); it!=EntryM.end(); it++){
		long hash_idx = it->first;
		double ra = it->second;
		int p = (int)(hash_idx/M);
		int q = (int)(hash_idx%M);
		int b_p = p/p_height;
		int b_q = q/q_height;
		int b_t_p = b_p/t_p_height;
		int b_t_q = b_q/t_q_height;
		Rblocks[b_t_p][b_t_q].push_back(RatingEntry(p,q,ra));
	}


	/*
	for (i = 0; i < BK_NUM; i++)
	{
		long row_sum = 0;
		printf("[%d]\t", i);
		for (j = 0; j < BK_NUM; j++)
		{
			printf("%ld\t", Rblocks[i][j].size());
			row_sum += Rblocks[i][j].size();
		}
		printf("{%ld}\n", row_sum );
	}
	**/

}
void initParas()
{
	partitionP(BK_NUM, N, PBlocks);
	partitionP(BK_NUM, M, QBlocks);
	int i, j;
	for (i = 0; i < BK_NUM; i++)
	{
		for (j = 0; j < PBlocks[i].ele_num; j++)
		{
			PBlocks[i].eles[j] = drand48() * 0.2;
		}
		for (j = 0; j < QBlocks[i].ele_num; j++)
		{
			QBlocks[i].eles[j] = drand48() * 0.2;
		}
	}
	for (i = 0; i < ITER_CAP; i++)
	{
		for (j = 0; j < TD_NUM; j++)
		{
			random_seq[i * TD_NUM + j] = j;
		}
	}
	for (i = 0; i < ITER_CAP; i++)
	{
		random_shuffle(random_seq + i * TD_NUM, random_seq + (i + 1)*TD_NUM );
	}
	printf("debug...\n");
	for(i = 0; i<10; i++){
		printf("%d\t", random_seq[i]);
	}

}
void allocCudaMem()
{
	HANDLE_ERROR(cudaMalloc((void**)&dev_PBlocks, sizeof(Block)* BK_NUM));
	HANDLE_ERROR(cudaMalloc((void**)&dev_QBlocks, sizeof(Block)* BK_NUM));
	int i = 0;
	for ( i = 0; i < BK_NUM; i++)
	{
		HANDLE_ERROR(cudaMalloc((void**) &(dev_PData[i]), sizeof(double) * (PBlocks[i].ele_num)));
		HANDLE_ERROR(cudaMalloc((void**) &(dev_QData[i]), sizeof(double) * (QBlocks[i].ele_num)));
	}

	HANDLE_ERROR(cudaMalloc((void**)&dev_seq, sizeof(int) * (TD_NUM * ITER_CAP)) );
	HANDLE_ERROR(cudaMalloc((void**)&dev_flag, sizeof(int) * (BK_NUM)) );

	int j =0;
	int idx = 0;
	for(i = 0; i<BT_NUM; i++){
		for(j = 0; j < BT_NUM; j++){
			idx = i*(BT_NUM)+j;
			entry_num[idx] = Rblocks[i][j].size();
			HANDLE_ERROR(cudaMalloc((void**)&(dev_rate_entries[idx]), sizeof(RatingEntry) * (entry_num[idx])) );
		}
	}
	for(i = 0; i<BT_NUM; i++){
		HANDLE_ERROR(cudaMalloc((void**)&(dev_p_cache[i]), sizeof(double) * (K)) );
		HANDLE_ERROR(cudaMalloc((void**)&(dev_q_cache[i]), sizeof(double) * (K)) );
	}

	//should be dynamic
	HANDLE_ERROR(cudaMemcpy( (dev_PBlocks), PBlocks, sizeof(Block) * (BK_NUM), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy( (dev_QBlocks), QBlocks, sizeof(Block) * (BK_NUM), cudaMemcpyHostToDevice));
	for (i = 0; i < BK_NUM; i++)
	{
		HANDLE_ERROR(cudaMemcpy( (dev_PData[i]), PBlocks[i].eles, sizeof(double) * (PBlocks[i].ele_num), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy( (dev_QData[i]), QBlocks[i].eles, sizeof(double) * (QBlocks[i].ele_num), cudaMemcpyHostToDevice));
	}


	HANDLE_ERROR(cudaMemcpy( (dev_seq), random_seq, sizeof(int) * (TD_NUM * ITER_CAP), cudaMemcpyHostToDevice));

////
	HANDLE_ERROR(cudaMemcpy( (dev_entry_num), entry_num, sizeof(int) * (RB_NUM), cudaMemcpyHostToDevice));
////

	for(i = 0; i<BT_NUM; i++){
		for(j = 0; j < BT_NUM; j++){
			idx = i*(BT_NUM)+j;
			if(Rblocks[i][j].size()> 0){
					RatingEntry* cache_re = (RatingEntry*)malloc(sizeof(RatingEntry)*Rblocks[i][j].size());
					int k = 0;
					for(k =0; k < Rblocks[i][j].size(); k++){
						cache_re[k] = Rblocks[i][j][k];
					}
					HANDLE_ERROR(cudaMemcpy( (dev_rate_entries[idx]), cache_re, sizeof(RatingEntry) * (Rblocks[i][j].size()), cudaMemcpyHostToDevice));
					free(cache_re);
			}
			
		}
	}



}
void freeCudaMem()
{
	int i, j;
	cudaFree(dev_PBlocks);
	cudaFree(dev_QBlocks);
	for ( i = 0; i < BK_NUM; i++)
	{
		cudaFree(dev_PData[i]);
		cudaFree(dev_QData[i]);
	}
	cudaFree(dev_seq);
	cudaFree(dev_flag);

	for(i = 0; i<BT_NUM; i++){
		for(j = 0; j < BT_NUM; j++){
			cudaFree(dev_rate_entries[i*BT_NUM+j]);
		}
	}
	for(i = 0; i<BT_NUM; i++){
		cudaFree(dev_p_cache[i]);
		cudaFree(dev_q_cache[i]);
	}
}
void partitionP(int portion_num, int line_num,  Block * block_arr)
{
	int i = 0;
	int height = (line_num+portion_num-1) / portion_num;
	//int last_height = N - (portion_num - 1) * height;

	for (i = 0; i < portion_num; i++)
	{
		block_arr[i].block_id = i;
		block_arr[i].data_age = 0;
		block_arr[i].height = height;
		int sta_idx = i * height;
		/*
		if ( i == portion_num - 1)
		{
			block_arr[i].height = last_height;
		}
		**/
		block_arr[i].sta_idx = sta_idx;
		block_arr[i].ele_num = block_arr[i].height * K;
		block_arr[i].eles = Malloc(double, block_arr[i].ele_num);
	}

}

int main(void)
{
	readTrainData();
	printf("readTrainData Fini\n");
	initParas();
	printf("initParas Fini\n");
	allocCudaMem();
	printf("allocCudaMem Fini\n");
	getchar();
	MFkernel <<< SM_NUM, 1>>>(dev_PBlocks, dev_QBlocks, dev_PData, dev_QData, dev_p_cache,dev_q_cache,dev_seq, dev_entry_num, dev_rate_entries, p_height, q_height, dev_flag, 0, yita, theta);
	printf("MFkernel Fini\n");
	cudaDeviceSynchronize();
	printf("cudaDeviceSynchronize Fini\n");
	freeCudaMem();
	printf("freeCudaMem Fini\n");
	//cudaDeviceReset();
	return 0;
}
