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
#include "Week2.h"

vector<vector<vector<float> > >BoxPooling(int PoolSize,  const vector<vector<vector<float> > >& InputBox)
{
    int BoxFilterCount = (InputBox).size();
    int BoxSize = (InputBox)[0].size()/PoolSize;
    vector<vector<vector<float> > > Box (BoxFilterCount,vector<vector<float>>(BoxSize, vector<float>(BoxSize, 0)));
    for(int i=0 ;i < BoxFilterCount; i++)
    {
        Box[i] = MaxPooling(0, PoolSize, (InputBox)[i], PoolSize);
    }

    return Box;

}
vector<vector<vector<float> > >BoxConvolution (const vector<vector<vector<float> > >& InputBox, const  vector<vector<vector<vector<float> > > >&Weights, const  vector<float>& Bias)
{
    int BoxFilterCount = (Bias).size();
    int InputChannelSize = (InputBox).size();
    int BoxSize = (InputBox)[0].size() - (Weights)[0][0].size()+1;
    vector<vector<vector<vector<float> > > >Box (BoxFilterCount ,vector<vector<vector<float>>>(InputChannelSize,vector<vector<float>>(BoxSize, vector<float>(BoxSize, 0))));
    for(int i=0 ;i < BoxFilterCount; i++)
    {
        for(int j=0 ;j < InputChannelSize; j++)
        {
            Box[i][j] = convolution(0,(InputBox)[j], (Weights)[i][j]);
        }
    }
    vector<vector<vector<float> > > Result(BoxFilterCount,vector<vector<float>>(BoxSize, vector<float>(BoxSize, 0)));
    for(int i=0;i<BoxFilterCount;i++)
    {
        for(int j =0; j<BoxSize;j++)
        {
            for(int k=0; k< BoxSize;k++)
            {
                for(int l = 0; l<InputChannelSize;l++)
                {
                   Result[i][j][k] +=Box[i][l][j][k]; 
                }
                Result[i][j][k] += Bias[i];
                //Result[i][j][k] = tanh(Result[i][j][k]); 
            }
        }
    }
    return Result;
}

vector<vector<vector<float> > >FullyConnectedBox(const vector<vector<vector<float> > >& InputBox, const vector<vector<vector<vector<float> > > >& Weights, const vector<float>& Bias)
{
    vector<vector<vector<float> > >Box = BoxConvolution(InputBox,Weights, Bias);
    int BoxFilterCount = Box.size();
    int BoxSize = Box[0].size();
    vector<vector<vector<float> > > Result(BoxFilterCount,vector<vector<float>>(BoxSize, vector<float>(BoxSize, 0)));
    
    for(int l=0;l<BoxSize;l++)
    {
        for(int k=0; k< BoxSize;k++)
        {
            for(int j =0; j<BoxFilterCount;j++)
            {
                Result[j][k][l] =ReLU(Box[j][k][l]); 

            }
        }
    }
    return Result;

}


int ImageIdentifer(string filename)
{
    //-----------------------------The Input Matrix is 28 X 28 ---------------------------------------------------------------------
    int InputChannelSize = 1;
    int imageSize = 28;
    string pixel;
    ifstream infile;
    infile.open (filename);
    vector<vector<vector<float> >>Input(InputChannelSize,vector<vector<float>>(imageSize, vector<float>(imageSize, 0)));
    for(int i=0 ;i<imageSize ;i++)
    {
        for(int j=0; j<imageSize;j++)
        {
            getline(infile,pixel);
            Input[0][i][j] = 1 - stof(pixel)/255;
        }
    }
    //printBox(Input);
    infile.close();
    //cout<<"Input Done"<<endl;
     //-----------------------------The First Convolution Filter Matrix is 20 X 5 X 5 (20 filters of size 5x5)----------------------------------------------------------
    int filterSize = 5;
    int filterCount = 20;

    vector<vector<vector<vector<float> > > >FirstFeatureMapBox(filterCount, vector<vector<vector<float> > >(InputChannelSize,vector<vector<float>>(filterSize, vector<float>(filterSize, 0))));
    string weight;
    string bias;
	infile.open ("conv1.txt");
	
	for(int i=0 ;i<filterCount ;i++)
    {
        for(int j=0; j<InputChannelSize;j++)
        {
            for(int k=0; k< filterSize; k++)
            {
                for(int l=0;l<filterSize;l++)
                {
                    getline(infile,weight);
                    FirstFeatureMapBox[i][j][k][l] = stof(weight);
                }
            }
            //printMatrix(FirstFeatureMapBox[i][j]);   
        }
		//printBox(FirstFeatureMapBox[i]);
    }

    vector<float> FirstBias (filterCount,0); 
    for(int l=0;l<filterCount;l++)
    {
        getline(infile,bias);
        FirstBias[l] = stof(bias);
    }
    //printVector(FirstBias);
    infile.close();
    //cout<<"First Weights Done"<<endl;
     //-----------------------------The First Layer Matrix is 20 X 24 X 24 (20 matrices of size 24x24 after convolution)--------------------------
    
    vector<vector<vector<float> > > FirstLayerBox = BoxConvolution( Input, FirstFeatureMapBox, FirstBias);
    //printMatrix(FirstLayerBox[0]);
    //printBox(FirstLayerBox);
    FirstBias.resize(0);
    Input.resize(0,vector<vector<float>>(0, vector<float>(0, 0)));
    FirstFeatureMapBox.resize(0, vector<vector<vector<float> > >(0,vector<vector<float>>(0, vector<float>(0, 0))));
    //cout<<"First Layer Done"<<endl;
    
    
    //-------------------The Second Layer Matrix is 20 X 12 X 12 (20 matrices of size 12x12 after max pooling in poolsize 2x2)---------------
    int PoolSize = 2;
    vector<vector<vector<float> > > SecondLayerBox = BoxPooling(PoolSize, FirstLayerBox) ;
    //printBox(SecondLayerBox);
    FirstLayerBox.resize(0,vector<vector<float>>(0, vector<float>(0, 0)));
    //cout<<"Second Layer Done"<<endl;

    //-----------------------------The Second Convolution Filter Matrix is 50 X 5 X 5 (50 filters of size 5x5)----------------------------------------------------------
    
    filterCount = 50;
    InputChannelSize = SecondLayerBox.size();
	infile.open ("conv2.txt");
    vector<vector<vector<vector<float> > > >SecondFeatureMapBox(filterCount, vector<vector<vector<float> > >(InputChannelSize,vector<vector<float>>(filterSize, vector<float>(filterSize, 0))));
    for(int i=0 ;i<filterCount ;i++)
    {
        for(int j=0; j<InputChannelSize;j++)
        {
            for(int k=0; k< filterSize; k++)
            {
                for(int l=0;l<filterSize;l++)
                {
                    getline(infile,weight);
                    SecondFeatureMapBox[i][j][k][l] = stof(weight);
                }
            }   
        }
    }

    vector<float>  SecondBias (filterCount,0); 
    for(int l=0;l<filterCount;l++)
    {
        getline(infile,bias);
        SecondBias[l] = stof(bias);
    }
    infile.close();
    //cout<<"Second Weights Done"<<endl;
    //-------------------The Third Layer Matrix is 50 X 8 X 8 (50 matrices of size 8x8 after convolution with SecondFeatureMapBox)-------------
    vector<vector<vector<float> > > ThirdLayerBox = BoxConvolution( SecondLayerBox, SecondFeatureMapBox, SecondBias);
    //printMatrix(ThirdLayerBox[0]);
    SecondBias.resize(0);
    SecondLayerBox.resize(0,vector<vector<float>>(0, vector<float>(0, 0)));
    SecondFeatureMapBox.resize(0, vector<vector<vector<float> > >(0,vector<vector<float>>(0, vector<float>(0, 0))));
    //cout<<"Third Layer Done"<<endl;

    //-------------------The Fourth Layer Matrix is 50 X 4 X 4 (50 matrices of size 4x4 after max pooling in poolsize 2x2)-------------------

    vector<vector<vector<float> > > FourthLayerBox = BoxPooling(PoolSize, ThirdLayerBox);
    //printMatrix(FourthLayerBox[0]);
    ThirdLayerBox.resize(0,vector<vector<float>>(0, vector<float>(0, 0)));
    //cout<<"Fourth Layer Done"<<endl;

    //----------------------The Third Convolution Filter Matrix is 500 X 4 X 4 (500 filters of size 4x4)----------------------------------------------------------
    
    filterSize = 4;
    filterCount = 500;
    InputChannelSize = FourthLayerBox.size();
	infile.open ("fc1.txt");
    vector<vector<vector<vector<float> > > >ThirdFeatureMapBox(filterCount, vector<vector<vector<float> > >(InputChannelSize,vector<vector<float>>(filterSize, vector<float>(filterSize, 0))));
    for(int i=0 ;i<filterCount ;i++)
    {
        for(int j=0; j<InputChannelSize;j++)
        {
            for(int k=0; k< filterSize; k++)
            {
                for(int l=0;l<filterSize;l++)
                {
                    
                    getline(infile,weight);
                    ThirdFeatureMapBox[i][j][k][l] = stof(weight);
                }
            }   
        }
    }

    vector<float>  ThirdBias (filterCount,0); 
    for(int l=0;l<filterCount;l++)
    {
        getline(infile,bias);
        ThirdBias[l] = stof(bias);
    }
    infile.close();
    //cout<<"Third Weights Done"<<endl;
    //-------------------The Fifth Layer Matrix is 500 X 1 X 1 (50 matrices of size 1x1 after convolution with ThirdFeatureMapBox)-------------
    
    vector<vector<vector<float> > > FifthLayerBox = FullyConnectedBox( FourthLayerBox, ThirdFeatureMapBox, ThirdBias);
    //printMatrix(FifthLayerBox[0]);
    //printMatrix(FifthLayerBox[1]); 
    ThirdBias.resize(0);
    FourthLayerBox.resize(0,vector<vector<float>>(0, vector<float>(0, 0)));
    ThirdFeatureMapBox.resize(0, vector<vector<vector<float> > >(0,vector<vector<float>>(0, vector<float>(0, 0))));
    //cout<<"Fifth Layer Done"<<endl;

    //----------------------The Fourth Convolution Filter Matrix is 10 X 1 X 1 (10 filters of size 1x1)----------------------------------------------------------
    
    filterSize = 1;
    filterCount = 10;
    InputChannelSize = FifthLayerBox.size();
	infile.open ("fc2.txt");
    vector<vector<vector<vector<float> > > >FourthFeatureMapBox(filterCount, vector<vector<vector<float> > >(InputChannelSize,vector<vector<float>>(filterSize, vector<float>(filterSize, 0))));
    for(int i=0 ;i<filterCount ;i++)
    {
        for(int j=0; j<InputChannelSize;j++)
        {
            for(int k=0; k< filterSize; k++)
            {
                for(int l=0;l<filterSize;l++)
                {
                    getline(infile,weight);
                    FourthFeatureMapBox[i][j][k][l] = stof(weight);
                }
            }   
        }
    }

    vector<float>  FourthBias (filterCount,0); 
    for(int l=0;l<filterCount;l++)
    {
        getline(infile,bias);
        FourthBias[l] = stof(bias);
    }
    //cout<<"Fourth Weights Done"<<endl;
    infile.close();
    //-------------------The Sixth Layer Matrix is 10 X 1 X 1 (10 matrices of size 1x1 after convolution with FourthFeatureMapBox)-------------
    
    vector<vector<vector<float> > > SixthLayerBox = BoxConvolution( FifthLayerBox, FourthFeatureMapBox, FourthBias);
    //printBox(SixthLayerBox);
    FourthBias.resize(0);
    FifthLayerBox.resize(0,vector<vector<float>>(0, vector<float>(0, 0)));
    FourthFeatureMapBox.resize(0, vector<vector<vector<float> > >(0,vector<vector<float>>(0, vector<float>(0, 0))));
    //cout<<"Sixth Layer Done"<<endl;

    vector<float>  Result(filterCount,0);
    for(int l=0;l<filterCount;l++)
    {
        Result[l] = SixthLayerBox[l][0][0];

    }
    SixthLayerBox.resize(0,vector<vector<float>>(0, vector<float>(0, 0)));
    //-------------------The Last Layer is SoftMax Layer in which there is vector of size 10. Index i corresponds with probability input image belongs to class i-------------
    vector<float> ProbabilityVector = softMax(Result);
    
    int digit = 0;
    float maxProbability = 0;
    for(int i =0; i< filterCount;i++)
    {
		if(ProbabilityVector[i]>maxProbability)
		{
			digit = i;
			maxProbability = ProbabilityVector[i];
		}
		cout<< " The probability of image being the number " + to_string(i) + " is " + to_string(ProbabilityVector[i]) + "." <<endl;
	}
	return digit;
    
    
}

int main(int argc, char** argv)
{
	vector<string> Word;
    for(int i =1 ;i<argc;i++)
    {
        Word.push_back(argv[i]);
    //cout << Word[i-1]<<endl;
    }

    if(argc > 1)
    {
        try
            {   string location1 = Word[0];
                cout << "The digit is " + to_string(ImageIdentifer(location1)) + "."<<endl;
            }
        catch(const exception FileNotFound) { cerr << "No such file exists. Please check file name of InputFile again" << endl;return 0;}

                
    }
    cout << "The digit is " + to_string(ImageIdentifer("1_new.txt")) + "."<<endl;
	cout << "The digit is " + to_string(ImageIdentifer("2_new.txt")) + "."<<endl;
}
