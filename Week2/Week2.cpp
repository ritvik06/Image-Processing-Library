#include<iostream>
#include<vector>
#include<cmath>
 #include<cblas.h>
#include<sstream>
#include<fstream>
#include<stdexcept>
#include<pthread.h>
#include<chrono>
#include<cstdlib>
//#include "Week2.h"
// #include "mkl.h"
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

void printMatrix(const vector<vector<float>>& Matrix);
void printArray(float* Array, int ArraySize);

void Vector_To_Matrix(const vector<vector<float> >& input,int Rowsize, int ColSize, float Matrix[])
{
	//Matrix = (float*)malloc(sizeof(float)*size*size);

	for(int i=0;i<Rowsize;i++){
		for(int j=0;j<ColSize;j++){
			//cout<<input[i][j]<<endl;
			//cout <<i*Rowsize+j<<endl;
			Matrix[i*ColSize+j]=input[i][j];
		}
	}
	//printArray(Matrix, Rowsize*ColSize);
	return;

}

vector<vector<float> > Matrix_To_Vector(int Rowsize, int ColSize, float Matrix[])
{
	//Matrix = (float*)malloc(sizeof(float)*size*size);
	vector<vector<float> >result(Rowsize, vector<float>(ColSize, 0));
	for(int i=0;i<Rowsize;i++){
		for(int j=0;j<ColSize;j++){
			result[i][j]=Matrix[i*ColSize+j];
			//cout<<"yhy"<<endl;
		}
	}
	return result;

}

vector<vector<float> > MatrixReLU(const vector<vector<float> >& input)
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

vector<vector<float> > MatrixTanh(const vector<vector<float> >& input)
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

vector<float> sigmoid(const vector<float>& input)
{
	vector<float> result;
	for(auto i = input.begin(); i != input.end(); ++i)
	{
		result.push_back(1/(1+exp(-1*(*i))));
	}
	return result;
}

vector<float> softMax(const vector<float>& input)
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

vector<vector<float> > padding(int pad,const vector<vector<float> >& input)
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


vector<vector<float> > convolution(int pad, const vector<vector<float> >& UnpaddedInput,const vector<vector<float> >& kernel)
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

vector<vector<float> > matrixMultiply(const vector<vector<float> >& A, const vector<vector<float> >& B)
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
vector<vector<float> > Mult_OpenBlas(vector<vector<float> >input1, vector<vector<float> >input2)
{
	float Matrix1[input1.size() * input1[0].size()]{} ;
	float Matrix2[input2.size() * input2[0].size()]{} ;
	float Matrix3[input1.size() * input2[0].size()]{} ;
	
	Vector_To_Matrix(input1,input1.size(),input1[0].size(),Matrix1);
	Vector_To_Matrix(input2,input2.size(),input2[0].size(),Matrix2);

	int m = input1.size();
	int n = input2[0].size();
	int k = input1[0].size();
	
	//printArray(Matrix1,m*k);
	//printArray(Matrix2,k*n);
	//printArray(Matrix3,m*n);
	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, m, n ,k, 1,Matrix1 ,k, Matrix2, n, 0 ,Matrix3, n);
	vector<vector<float> >Result = Matrix_To_Vector(input1.size(),input2[0].size(),Matrix3);
	//printArray(Matrix3,m*n);
	
	return Result;

}

// vector<vector<float> > Mult_MKL(vector<vector<float> >input1, vector<vector<float> >input2)
// {
// 	float Matrix1[input1.size() * input1[0].size()]{} ;
// 	float Matrix2[input2.size() * input2[0].size()]{} ;
// 	float Matrix3[input1.size() * input2[0].size()]{} ;
	
// 	Vector_To_Matrix(input1,input1.size(),input1[0].size(),Matrix1);
// 	Vector_To_Matrix(input2,input2.size(),input2[0].size(),Matrix2);

// 	int m = input1.size();
// 	int n = input2[0].size();
// 	int k = input1[0].size();
	
// 	//printArray(Matrix1,m*k);
// 	//printArray(Matrix2,k*n);
// 	//printArray(Matrix3,m*n);
// 	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, m, n ,k, 1,Matrix1 ,k, Matrix2, n, 0 ,Matrix3, n);
// 	vector<vector<float> >Result = Matrix_To_Vector(input1.size(),input2[0].size(),Matrix3);
// 	//printArray(Matrix3,m*n);
	
// 	return Result;
// }


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

vector<vector<float> > matrixMultiplyThreading(const vector<vector<float> >& A, const vector<vector<float> >& B,int numberOfThreads)
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

vector<vector<float> > scopedMatrix(const vector<vector<float> >& input,int scopedMatrixSize, int Row, int Column)
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

vector<vector<float> > squeezeMatrixToColumn(const vector<vector<float> >& input)
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

vector<vector<float> > flattenMatrixToRow(const vector<vector<float> >& input)
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

vector<vector<float> > Toeplitzise(const vector<vector<float> >& input, int KernelSize)
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

vector<vector<float> > convolutionMM(int pad, const vector<vector<float> >& UnpaddedInput, const vector<vector<float> >& kernel, int method)
{
   vector<vector<float> > input = padding(pad,UnpaddedInput);			
   unsigned InputSize = input.size();
   unsigned KernelSize = kernel.size();
   unsigned ProcessedKernelColumnSize = KernelSize*KernelSize; // It will be a column vector only;
   unsigned ResultSize = InputSize-KernelSize+1;
   vector<vector<float> > ProcessedInput = Toeplitzise(input,KernelSize);
   vector<vector<float> > ProcessedKernel = squeezeMatrixToColumn(kernel);
   vector<vector<float> > ProcessedResult;
   if(method == 0)
   {
   	ProcessedResult = matrixMultiply(ProcessedInput,ProcessedKernel);
   }
  else if(method == 1)
   {
   	ProcessedResult = Mult_OpenBlas(ProcessedInput,ProcessedKernel);
   }
   // else if(method == -1)
   // {
   // 	ProcessedResult = Mult_MKL(ProcessedInput,ProcessedKernel);
   // }
   else
   {
   	throw invalid_argument("Method Not Correct");
   }
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

vector<vector<float> > AveragePooling(int pad,int PoolSize, const vector<vector<float> >& UnpaddedInput)
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

vector<vector<float> > MaxPooling(int pad,int PoolSize, const vector<vector<float> >& UnpaddedInput, int Stride)
{
   vector<vector<float> > input = padding(pad,UnpaddedInput);			
   unsigned InputSize = input.size();
   unsigned ResultSize = InputSize-PoolSize+1;
   vector<vector<float> >result(ResultSize, vector<float>(ResultSize, 0));
   for(int i = 0; i < ResultSize; i+=Stride)
   {
		for(int j = 0; j< ResultSize;j+=Stride)
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

void printMatrix(const vector<vector<float>>& Matrix)
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
void printArray(float* Array, int ArraySize)
{
	// Printing the output on CMDLINE
	for(int j =0;j<ArraySize;j++)
	{
		cout << Array[j];
		cout << " ";
	}
	cout << endl;
}

void printVector(const vector<float>& Vector)
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
		cout<<endl;
	return;
}

void MatrixMultiplyTimedTesterPthreads(const vector<vector<float>>& A,int ThreadCount, bool printResult)
{
		auto start = std::chrono::high_resolution_clock::now();
		vector<vector<float>> result = matrixMultiplyThreading(A,A,ThreadCount);
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "Time taken for multiplication of 2 square matrices of size: "<< A.size()<< " is "  << elapsed.count() << "s using " << ThreadCount <<" threads."<<endl;
		if(printResult)
			printMatrix(result);
}

void MatrixMultiplyTimedTester(const vector<vector<float>>& A, bool printResult,int method)
{
		ofstream myfile;
		myfile.open ("output.txt", std::ios_base::app);
		auto start = std::chrono::high_resolution_clock::now();
		vector<vector<float>> result;
		string package;
		if(method == 1)
			{
				result = Mult_OpenBlas(A,A);
				package = "OpenBlas";
			}
		/*else if (method == -1)
			{
				result = Mult_MKL(A,A);
				package = "Intel MKL";
			}*/
		else 
			{
				result = matrixMultiply(A,A);
				package = "pthreads";
			}
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		myfile << A.size()<< " "  << elapsed.count()*(1000000) << " "<<endl;
		std::cout << "Time taken for multiplication of 2 square matrices of size: "<< A.size()<< " is "  << elapsed.count()*(1000000) << " microseconds using " + package <<endl;
		if(printResult)
			printMatrix(result);
		myfile.close();
}
void MatrixMultiplyData(const vector<vector<float>>& A, bool printResult,int method,int trials)
{
	myfile.open ("Data.txt", std::ios_base::app);
	for(int i =0;i<trials;i++)
	{
		auto start = std::chrono::high_resolution_clock::now();
		vector<vector<float>> result;
		if(method == 1)
			{
				result = Mult_OpenBlas(A,A);
			}
		/*else if (method == -1)
			{
				result = Mult_MKL(A,A);
				package = "Intel MKL";
			}*/
		else 
			{
				result = matrixMultiply(A,A);
			}
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		myfile << A.size()<< " "  << elapsed.count()*(1000000) << " "<<endl;
		
	}
	myfile.close();
}

int main(int argc, char** argv)
{
	vector<string> Word;
	vector<float> input_vector;
	for(int i =1 ;i<argc;i++)
	{
		Word.push_back(argv[i]);
	//cout << Word[i-1]<<endl;
	}

	if(argc > 1)
	{
		if(Word[0]=="convolution" || Word[0]=="convolutionMM" || Word[0]=="MMOpenBlas")
		{	
			int pad_size =0;
			try
			{
				pad_size = stoi(Word[1]);
				if(pad_size < 0)
					throw invalid_argument("recieved negative value");
			}
			catch(const invalid_argument NotANumber){cerr << "Pad Size must be a number" << endl;return 0;}

			string location1 = Word[2];
			int matrix1_numrows = 0;
			try
			{
				matrix1_numrows = stoi(Word[3]);
				if (matrix1_numrows <= 0)
					throw invalid_argument("recieved non postive value");
			}
			catch(const invalid_argument NotANumber){cerr << "Row Number must be a positive number" << endl;return 0;}

			string location2 = Word[4];
			int matrix2_numrows = 0;
			try
			{
				matrix2_numrows = stoi(Word[5]);
				if (matrix2_numrows <= 0)
					throw invalid_argument("recieved non postive value");
			}
			catch(const invalid_argument NotANumber){ cerr << "Row Number must be a positive number" << endl;return 0;}

			vector<vector<float>> matrix1(matrix1_numrows,vector<float>(matrix1_numrows,0));
			vector<vector<float>> matrix2(matrix2_numrows,vector<float>(matrix2_numrows,0));

			try
			{
				ifstream input1(location1);
				if(input1.is_open())
				{
					for(int i=0; i<matrix1_numrows; i++)
					{
						for(int j=0; j<matrix1_numrows; j++)
						{
							string str1;
							getline(input1,str1, '\n');
							istringstream iss(str1);
							string temp;
							iss >> temp;
							matrix1[j][i]=stof(temp);
						}
					}
					cout << "Input Matrix is -" <<endl;
					printMatrix(matrix1);
				}
				else
				 throw exception();
			}
			catch(const exception FileNotFound) { cerr << "No such file exists. Please check file name of InputFile again" << endl;return 0;}

			try
			{
				ifstream input2(location2);
				if(input2.is_open())
				{
					for(int k=0; k<matrix2_numrows; k++)
					{
						for(int l=0; l<matrix2_numrows; l++)
						{
							string str2;
							getline(input2,str2, '\n');
							istringstream iss(str2);
							string temp;
							iss >> temp; 
							matrix2[l][k] = stof(temp);
						}
					}
					cout << "Kernel Matrix is -" <<endl;
					printMatrix(matrix2);
				}
				else 
					throw exception();
			}
			catch(const exception FileNotFound) { cerr << "No such file exists. Please check file name of Kernel again" << endl;return 0;}
			
			if(Word[0]=="convolution")
			{
				//cout << "here" << endl;
				vector<vector<float>> result = convolution(pad_size,matrix1,matrix2);
				//cout << "here" << endl;
				cout << "Result Matrix is -" <<endl;
				printMatrix(result);
			}
			else if(Word[0]=="convolutionMM")
			{
				//cout << "hereaswell" << endl;
				vector<vector<float>> result = convolutionMM(pad_size,matrix1,matrix2,0);
				//cout << "hereaswell" << endl;	
				cout << "Result Matrix is -" <<endl;
				printMatrix(result);
			}

			else if(Word[0]=="MMOpenBlas")
			{
				//cout << "hereaswell" << endl;
				vector<vector<float>> result = convolutionMM(pad_size,matrix1,matrix2,1);
				//cout << "hereaswell" << endl;	
				cout << "Result Matrix is -" <<endl;
				printMatrix(result);
			}

			else if(Word[0]=="MMIntel_mkl")
			{
				//cout << "hereaswell" << endl;
				vector<vector<float>> result = convolutionMM(pad_size,matrix1,matrix2,-1);
				//cout << "hereaswell" << endl;	
				cout << "Result Matrix is -" <<endl;
				printMatrix(result);
			}
			//else if (Word[0]=="MMOpenBlas")
		
		}
		else if(Word[0]=="tanh" || Word[0]=="ReLU")
		{
			string location = Word[1];
			int matrix_numrows=0;
			try
			{
				matrix_numrows = stoi(Word[2]);
				if (matrix_numrows <= 0)
					throw invalid_argument("recieved non postive value");
			}
			catch(const invalid_argument NotANumber){ cerr << "Row Number must be a positive number" << endl;return 0;}

			vector<vector<float>> input_matrix(matrix_numrows,vector<float>(matrix_numrows,0));
			try
			{
				ifstream input(location);
				if(input.is_open())
				{
					for(int a=0; a<matrix_numrows; a++)
					{		
						for(int b=0; b<matrix_numrows; b++)
						{
							string str;
							getline(input,str);
							istringstream iss(str);
							string temp;
							iss >> temp;
							input_matrix[b][a] = stof(temp);
						}
					}
					cout << "Input Matrix is -" <<endl;
					printMatrix(input_matrix);
				}
				else 
					throw exception();
			}
			catch(const exception FileNotFound) { cerr << "No such file exists. Please check file name of InputMatrix again" << endl; return 0;}

			if(Word[0]=="tanh")
			{
				vector<vector<float>> result = MatrixTanh(input_matrix);
				cout << "Result Matrix is -" <<endl;
				printMatrix(result);
			}
			else if(Word[0]=="ReLU")
			{
				vector<vector<float>> result = MatrixReLU(input_matrix);
				cout << "Result Matrix is -" <<endl;
				printMatrix(result);
			}
		}

		else if(Word[0]=="sigmoid" || Word[0]=="softMax")
		{
			string location=Word[1];
			try
			{
				ifstream input(location);
				vector<float> input_vector;
				if(input.is_open())
				{
					string str;
					getline(input,str);
					istringstream iss(str);
					do
					{
						float temp;
						iss >> temp;
						input_vector.push_back(temp);
					}while(iss);
					cout << "Input vector is " << endl;
					printVector(input_vector);
					if(Word[0]=="sigmoid")
					{
						vector<float> result = sigmoid(input_vector);
						cout << "Result Vector is -" <<endl;
						printVector(result);
					}	
					else if(Word[0]=="softMax")
					{
						vector<float> result = softMax(input_vector);
						cout << "Result Vector is -" <<endl;
						printVector(result);
					}
				}
			} 
			catch(const exception FileNotFound) { cerr << "No such file exists. Please check file name of Input Vector again" << endl;return 0;}
		}
		
		else
		{
			cout << "Command Not Identified. Please Type One of {convolution, convolutionMM, sigmoid, softMax,tanh, ReLU}" << endl;
		}
	}
//	return 0;
//int main(){

	vector<vector<float>> inputMatrix = {{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5}};
	//vector<vector<float>> kernel = {{1,-2,0},{-1,2,-1},{1,2,0}};
	vector<vector<float>> kernel = {{0,0,0},{0,1,0},{0,0,0}};
	//printMatrix(inputMatrix);
	//printMatrix(kernel);
	vector<float> inputVector = {-5,-4,-3,-2,-1,0,1,2,3,4,5};
	vector<float> Size5 = {1,2,3,4,5};
	vector<float> Size10 = {1,2,3,4,5,6,7,8,9,10}; 
	vector<float> Size20 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	vector<float> Size30 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
	vector<float> Size40 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40};
	vector<float> Size50 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50};
	vector<float> Size60 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60};
	vector<vector<float>> Size5x5 = {Size5,Size5,Size5,Size5,Size5};
	vector<vector<float>> Size10x10 = {Size10,Size10,Size10,Size10,Size10,Size10,Size10,Size10,Size10,Size10};
	vector<vector<float>> Size20x20 = {Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20,Size20};
	vector<vector<float>> Size30x30 = {Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30,Size30};
	vector<vector<float>> Size40x40 = {Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40,Size40};
	vector<vector<float>> Size50x50 = {Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50,Size50};
	vector<vector<float>> Size60x60 = {Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60,Size60};
	
	//MatrixMultiplyTimedTesterPthreads(Size60x60,10,true);
	//MatrixMultiplyTimedTesterPthreads(Size10x10,true);
	//MatrixMultiplyTimedTesterPthreads(Size20x20,true);
	//printMatrix(Size10x10);
	int method = 1;
	MatrixMultiplyTimedTester(Size10x10,false,method);
	MatrixMultiplyTimedTester(Size20x20,false,method);
	MatrixMultiplyTimedTester(Size30x30,false,method);
	MatrixMultiplyTimedTester(Size40x40,false,method);
	MatrixMultiplyTimedTester(Size50x50,false,method);
	MatrixMultiplyTimedTester(Size60x60,false,method);
	
	method = 0;
	MatrixMultiplyTimedTester(Size10x10,false,method);
	MatrixMultiplyTimedTester(Size20x20,false,method);
	MatrixMultiplyTimedTester(Size30x30,false,method);
	MatrixMultiplyTimedTester(Size40x40,false,method);
	MatrixMultiplyTimedTester(Size50x50,false,method);
	MatrixMultiplyTimedTester(Size60x60,false,method);
	//Mult_OpenBlasMKL(Size10x10,Size10x10,Matrix3,10,10);
	//cout<< *(Matrix4	 + 4)<<endl;
	//printMatrixPointer(Matrix4,10);

	//free(Matrix4);	


	//MatrixMultiplyTimedTester(Size40x40,ThreadCount,true);
	//MatrixMultiplyTimedTester(Size50x50,ThreadCount,true);
	//MatrixMultiplyTimedTester(Size60x60,ThreadCount,true);
	// printMatrix(A);
	// printMatrix(flattenMatrixToRow(A));
	// printMatrix(squeezeMatrixToColumn(A));
	// printMatrix(B);
	// // printMatrix(matrixMultiply(A,B));
	// printVector(inputVector);
	// printVector(sigmoid(inputVector));
	// printVector(softMax(inputVector));
	// printMatrix(kernel);
	// vector<vector<float>> result = AveragePooling(0,1,inputMatrix);
	// vector<vector<float>> result = AveragePooling(1,2,inputMatrix);
	// vector<vector<float>> result = MaxPooling(0,1,inputMatrix);
	// printMatrix(padding(1, inputMatrix));
	// vector<vector<float>> result = MaxPooling(1,2,inputMatrix);
	// printMatrix(A);
	// vector<vector<float>> result = MatrixReLU(inputMatrix);
	// vector<vector<float>> result = MatrixTanh(inputMatrix);
	// vector<vector<float>> result = MaxPooling(0,1,inputMatrix);
	// printMatrix(inputMatrix);
	// printMatrix(Toeplitzise(inputMatrix,3));
	// printMatrix(scopedMatrix(inputMatrix,5,0,0));
	// vector<vector<float>> result2 = convolutionMM(0,inputMatrix,kernel);
	// printMatrix(result2);
	return 0;
}
