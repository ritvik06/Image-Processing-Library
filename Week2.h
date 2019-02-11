#include<vector>
#include<cmath>
#include<cblas.h>
#include<sstream>
#include<fstream>
#include<stdexcept>
#include<pthread.h>
#include<chrono>
#include<cstdlib>
//#include "mkl.h"
//#include <bit/stdc++.h>
//#include<sstring>

using namespace std;

//#define NUM_THREADS 3
#define MAX_MATRIX_SIZE 100
#define SIZE30 30
vector<vector<float>> GResult  = vector<vector<float> > (MAX_MATRIX_SIZE, vector<float>(MAX_MATRIX_SIZE, 0));

float ReLU(float num)
{
	if(num>0)
	{
		return num;
	}

	else
	{
		return 0;
	}
}

void printMatrix(vector<vector<float>> Matrix);

float* Vector_To_Matrix(vector<vector<float> > input,int size)
{
	float *Matrix;

	Matrix = (float*)malloc(sizeof(float)*size*size);

	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			Matrix[i*size+j]=input[i][j];
		}
	}

	return Matrix;
}

vector<vector<float> > MatrixReLU(vector<vector<float> > input)
{
   int size = input.size();
   vector<vector<float> > result (size, vector<float>(size, 0));
   for(int i = 0; i < size; i++)
   {
		for(int j = 0; j< size;j++)
		{
			result[i][j] = ReLU(input[i][j]);
		}
	}	
   return result;
}	

vector<vector<float> > MatrixTanh(vector<vector<float> > input)
{
   unsigned size = input.size();
   vector<vector<float> > result (size, vector<float>(size, 0));
   for(int i = 0; i < size; i++)
   {
		for(int j = 0; j< size;j++)
		{
			result[i][j] = tanh(input[i][j]);
		}
	}
   return result;
}	

vector<float> sigmoid(vector<float> input)
{
	vector<float> result;
	for(auto i = input.begin(); i != input.end(); ++i)
	{
		result.push_back(1/(1+exp(-1*(*i))));
	}
	return result;
}

vector<float> softMax(vector<float> input)
{
	float sum=0;
	vector<float> result;

	for(auto i = input.begin(); i != input.end(); ++i)
	{
		sum+=exp(*i);
	}
	for(auto i = input.begin(); i != input.end(); ++i)
	{
		result.push_back(exp(*i)/sum);
	}
	return result;
}

vector<vector<float> > padding(int pad,vector<vector<float> > input)
{
   unsigned size = input.size()+2*pad;
   vector<vector<float> >result(size, vector<float>(size, 0));
   for(int i = 0; i < size; i++)
   {
		for(int j = 0; j< size;j++)
		{
			if(i>= pad && i< size-pad && j>= pad && j< size-pad)
			{
				result[i][j] = input[i-pad][j-pad];
			}
			else
			{
				result[i][j] = 0;
			}
		}
	}
   return result;
}


vector<vector<float> > convolution(int pad, vector<vector<float> > UnpaddedInput, vector<vector<float> > kernel)
{
   vector<vector<float> > input = padding(pad,UnpaddedInput);			
   unsigned InputSize = input.size();
   unsigned KernelSize = kernel.size();
   unsigned ResultSize = InputSize-KernelSize+1;
   vector<vector<float> > result (ResultSize, vector<float>(ResultSize, 0));
   for(int i = 0; i < ResultSize; i++)
   {
		for(int j = 0; j< ResultSize;j++)
		{
			float sum = 0;
			for(int k = 0; k<KernelSize;k++)
			{
				for(int l = 0; l<KernelSize;l++)
				{
					sum+= input[i+k][j+l]*kernel[k][l];
				}
			}
			result[i][j] = sum;
		}
	}
   return result;
}

float* Mult_OpenBlas(vector<vector<float> > input1, vector<vector<float> > input2,float *AB,int size1,int size2)
{
	float *Matrix1,*Matrix2;

	Matrix1 = Vector_To_Matrix(input1,size1);
	Matrix2 = Vector_To_Matrix(input2,size2);


	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
		size1,size2,size1,
		1,Matrix1,size1,Matrix2,size2,
		0 ,AB,size2);


	return AB;

}

float* Mult_MKL(vector<vector<float> > input1, vector<vector<float> > input2,float *AB,int size1,int size2)
{
	float *Matrix1,*Matrix2;

	Matrix1 = Vector_To_Matrix(input1,size1);
	Matrix2 = Vector_To_Matrix(input2,size2);


	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
		size1,size2,size1,
		1,Matrix1,size1,Matrix2,size2,
		0 ,AB,size2);


	return AB;

}

vector<vector<float> > matrixMultiply(vector<vector<float> > A, vector<vector<float> > B)
{
	unsigned ARowSize = A.size();
	unsigned AColumnSize = A[0].size();
	unsigned BRowSize = B.size();
	unsigned BColumnSize = B[0].size();
	vector<vector<float> >result(ARowSize, vector<float>(BColumnSize, 0));
	if(AColumnSize==BRowSize)
	{
		for(int i = 0; i < ARowSize; i++)
   		{
			for(int j = 0; j< BColumnSize;j++)
			{
				float sum = 0;
				for(int k = 0; k<AColumnSize;k++)
				{
					sum += A[i][k]*B[k][j];
				}
				result[i][j] = sum;
			}
		}
	}
	return result;
}

typedef struct ThreadMMParams {
    vector<vector<float>> matrix1;
    vector<vector<float>> matrix2;
    unsigned ThreadID;
    unsigned matrix1_numrows;
    unsigned matrix2_numrows;
    unsigned matrix1_numcols;
    unsigned matrix2_numcols;
    unsigned TotalThreads;
}ThreadMMParams;

void *MM(void *param) {
   ThreadMMParams *Parameters = (ThreadMMParams *) param; // the structure that holds our data
//cout << "Reached Here" <<endl;
  for (int i = Parameters->ThreadID; i < Parameters->matrix1_numrows; i+=(Parameters->TotalThreads))
   {
		for (int j = 0; j < Parameters->matrix2_numcols; j++)
		{
   			int sum = 0;
			for(int k = 0; k < Parameters->matrix1_numcols ; k++)
			{
      			sum += (Parameters->matrix1)[i][k] * (Parameters->matrix2)[k][j];
			//cout <<(Parameters->matrix1)[i][k] * (Parameters->matrix2)[k][j]<<endl;
   			}
			GResult[i][j] = sum;
//			cout <<(Parameters->result)[i][j];
		}
   }
	//param = (void*) Parameters;
//	printMatrix(Parameters->result);
   //pthread_exit(0);
}

vector<vector<float> > matrixMultiplyThreading(vector<vector<float> > A, vector<vector<float> > B,int numberOfThreads)
{
//	cout << "Started At TOP" << endl;
	unsigned ARowSize = A.size();
	unsigned AColumnSize = A[0].size();
	unsigned BRowSize = B.size();
	unsigned BColumnSize = B[0].size();
	vector<vector<float> >result(ARowSize, vector<float>(BColumnSize, 0));
	if(AColumnSize==BRowSize)
	{
		pthread_t threads[numberOfThreads]; 
		for(int i = 0; i < numberOfThreads; i++ )
		{
			ThreadMMParams *Parameters = (ThreadMMParams *) malloc(sizeof(ThreadMMParams));
			Parameters->ThreadID = i;
			Parameters->matrix1_numrows = A.size();
			Parameters->matrix2_numrows = B.size();
			Parameters->matrix1_numcols = A[0].size();
			Parameters->matrix2_numcols = B[0].size();
			Parameters->matrix1 = A;
			Parameters->matrix2 = B;
			Parameters->TotalThreads = numberOfThreads;
		//	Parameters->result = result;
  //  		cout <<"Inside MMT"<<endl;
		pthread_create(&threads[i],NULL,MM,(void *)Parameters); /* Created Structs for Thread Data*/
		}

		for(int i = 0; i<numberOfThreads; i++ )
		{
			pthread_join(threads[i], NULL); 
		}
		for (int i = 0; i<ARowSize ;i++)
		{
			for(int j=0 ; j<BColumnSize ; j++)
			{
				result[i][j] = GResult[i][j];
			}
		}
		//printMatrix(result);
	}
	return result;
}

vector<vector<float> > scopedMatrix(vector<vector<float> > input,int scopedMatrixSize, int Row, int Column)
{			
	vector<vector<float> >scopedMatrix(scopedMatrixSize, vector<float>(scopedMatrixSize, 0));
	for(int i = 0; i < scopedMatrixSize; i++)
	{
		for(int j = 0; j< scopedMatrixSize;j++)
		{
			scopedMatrix[i][j] = input[Row+i][Column+j];
		}
	}
	return scopedMatrix;
}

vector<vector<float> > squeezeMatrixToColumn(vector<vector<float> > input)
{			
   unsigned InputSize = input.size();
   unsigned ColumnSize = InputSize*InputSize;
   vector<vector<float> >columnOfMatrix(ColumnSize,vector<float>(1, 0));
   for(int i = 0; i < ColumnSize; i++)
	{
		columnOfMatrix[i][0]=input[i/InputSize][i%InputSize];
	}
	return columnOfMatrix;
}

vector<vector<float> > flattenMatrixToRow(vector<vector<float> > input)
{			
   unsigned InputSize = input.size();
   unsigned RowSize = InputSize*InputSize;
   vector<vector<float> >rowOfMatrix(1,vector<float>(RowSize, 0));
   for(int i = 0; i < RowSize; i++)
	{
		rowOfMatrix[0][i]=input[i/InputSize][i%InputSize];
	}
	return rowOfMatrix;
}

vector<vector<float> > Toeplitzise(vector<vector<float> > input, int KernelSize)
{
   unsigned InputSize = input.size();
   unsigned ProcessedInputRowSize = KernelSize*KernelSize;
   unsigned ResultSize = InputSize-KernelSize+1;
   unsigned ProcessedInputColumnSize = ResultSize*ResultSize;
   unsigned RowNumber = 0; // Row Number for ProcessedInputMatrix
   vector<vector<float> >ProcessedInput(ProcessedInputColumnSize, vector<float>(ProcessedInputRowSize, 0));
   for(int i = 0; i < ResultSize; i++)
	{
		for(int j = 0; j< ResultSize;j++)
		{
			vector<vector<float> >CurrentRow = flattenMatrixToRow(scopedMatrix(input,KernelSize,i,j));
			for(int k=0;k<CurrentRow[0].size();k++)
			{
				ProcessedInput[RowNumber][k] = CurrentRow[0][k];
			}
			RowNumber++;
		}
	}
	return ProcessedInput;
}

vector<vector<float> > convolutionMM(int pad, vector<vector<float> > UnpaddedInput, vector<vector<float> > kernel)
{
   vector<vector<float> > input = padding(pad,UnpaddedInput);			
   unsigned InputSize = input.size();
   unsigned KernelSize = kernel.size();
   unsigned ProcessedKernelColumnSize = KernelSize*KernelSize; // It will be a column vector only;
   unsigned ResultSize = InputSize-KernelSize+1;
   vector<vector<float> > ProcessedInput = Toeplitzise(input,KernelSize);
   vector<vector<float> > ProcessedKernel = squeezeMatrixToColumn(kernel);
   vector<vector<float> > ProcessedResult = matrixMultiply(ProcessedInput,ProcessedKernel);
   vector<vector<float> > result(ResultSize, vector<float>(ResultSize, 0));
   for(int i = 0; i < ResultSize; i++)
	{
		for(int j = 0; j< ResultSize;j++)
		{
			result[i][j] = ProcessedResult[i*ResultSize+j][0];
		}
	}
	return result;
}

vector<vector<float> > AveragePooling(int pad,int PoolSize,  vector<vector<float> > UnpaddedInput)
{
   vector<vector<float> > input = padding(pad,UnpaddedInput);			
   unsigned InputSize = input.size();
   unsigned ResultSize = InputSize-PoolSize+1;
   vector<vector<float> >result(ResultSize, vector<float>(ResultSize, 0));
   for(int i = 0; i < ResultSize; i++)
   {
		for(int j = 0; j< ResultSize;j++)
		{
			float sum = 0;
			for(int k = 0; k<PoolSize;k++)
			{
				for(int l = 0; l<PoolSize;l++)
				{
					sum+= input[i+k][j+l];
				}
			}
		result[i][j] = sum/(PoolSize*PoolSize);
		}
	}
	return result;
}

vector<vector<float> > MaxPooling(int pad,int PoolSize,  vector<vector<float> > UnpaddedInput)
{
   vector<vector<float> > input = padding(pad,UnpaddedInput);			
   unsigned InputSize = input.size();
   unsigned ResultSize = InputSize-PoolSize+1;
   vector<vector<float> >result(ResultSize, vector<float>(ResultSize, 0));
   for(int i = 0; i < ResultSize; i++)
   {
		for(int j = 0; j< ResultSize;j++)
		{
			float max = input[i][j];
			for(int k = 0; k<PoolSize;k++)
			{
				for(int l = 0; l<PoolSize;l++)
				{
					if(input[i+k][j+l]>max)
					{
						max = input[i+k][j+l]; 
					}
				}
			}
			result[i][j] = max;
		}
	}
	return result;
}

void printMatrix(vector<vector<float>> Matrix)
{
	// Printing the output on CMDLINE
	for(int i =0;i<Matrix.size();i++)
	{
		for(int j =0;j<Matrix[0].size();j++)
		{
			cout << Matrix[i][j];
			cout << " ";
		}
		cout << endl;
	}
}

void printVector(vector<float> Vector)
{
	// Printing the output on CMDLINE
	for(int j =0;j<Vector.size();j++)
	{
		cout << Vector[j];
		cout << " ";
	}
	cout << endl;
}
void printMatrixPointer(float *Matrix,int size)
{
	for(int j =0;j<size*size; j++)
		{
			if(j%size==0)
				cout<<endl;
			float x = Matrix[j];
			cout << x ;
			cout << " ";
		}
	return;
}

void MatrixMultiplyTimedTesterPthreads(vector<vector<float>> A,int ThreadCount, bool printResult)
{
		auto start = std::chrono::high_resolution_clock::now();
		vector<vector<float>> result = matrixMultiplyThreading(A,A,ThreadCount);
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "Time taken for multiplication of 2 square matrices of size: "<< A.size()<< " is "  << elapsed.count() << "s using " << ThreadCount <<" threads."<<endl;
		if(printResult)
			printMatrix(result);
}

void MatrixMultiplyTimedTester(vector<vector<float>> A, bool printResult)
{
		ofstream myfile;
		myfile.open ("output.txt", std::ios_base::app);
		auto start = std::chrono::high_resolution_clock::now();
		float *Matrix3;
		Matrix3 = (float*)malloc(sizeof(float)*10*10);
		Mult_MKL(A,A,Matrix3,A.size(),A.size());
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		myfile << A.size()<< " "  << elapsed.count()*(1000000) << " "<<endl;
		if(printResult)
			printMatrixPointer(Matrix3,A.size());
		free(Matrix3);
		myfile.close();
}