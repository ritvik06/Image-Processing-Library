\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage[latin1]{inputenc}

\title{Optimizing Matrix Multiplication}
\author{Vasu Jain Ritvik Vij}
\date{29/01/19}

\begin{document}
\maketitle

\section{Subtask 2}

This subtask involved improving our Linear Algebra operations such as matrix multiplication using pthreads and commercial libraries such as multi-threading,
OpenBlas and Intel MKL.

\begin{itemize}
\item Using pthreads - if we have a N${\times}$N Matrix, suppose we use k threads for the matrix multiplication every k rows is applied to the same 
pthread, a thread contains a row which can be multiplied with the other matrix.
The pthread is a much faster implementation than the normal matrix multiplication. Using the toeplitz matrix, the matrix multiplication can be used to implement the convolution of a matrix with a filter.

\item Using OpenBlas - OpenBlas is a linear algebra library for fast implementation of linear algebra operations, sgemm function was used to implement matrix multiplication and is considerably fast for larger sizes of matrix upto approx 800${\times}$800.

\item Using Intel MKL - MKL linear algebra library is built upon the OpenBlas library but is considerably faster than OpenBlas and Pthreads,approximately 19 times faster for a matrix of size 200${\times}$200, the function used is the same sgemm(for operations on float matrices).

\end{itemize}

A GNU Plot was made to compare the time taken to perform this operation using pthreads, OpenBlas and Intel MKL.\\
\\
Further usage of convolution in the LeNet architecture for the digit recognition
software has been done using our very own pthread implementation.\\
\\
\\
\\
\\
\\
\subsection{Steps to run}
\\
\subsubsection{Pthreads}
./Week2 convolutionMM padsize InputMatrix.txt numrows1 Kernel.txt numrows2
\subsubsection{OpenBlas}
\begin{itemize}
\item g++ -std=c++11 -o Week2 Week2.cpp -I
/"C:/OpenBLAS-0.2.20/OpenBLAS-0.2.20"/OpenBLAS/include/ -L/your\_path/OpenBLAS/lib -lopenblas -lm -lpthread
\item ./Week2 MMOpenBlas padsize InputMatrix.txt numrows1 Kernel.txt numrows2
\end{itemize}
\subsubsection{Intel MKL}
\begin{itemize}
\item Inside Intel C/C++ Compiler
 your\_path = C:/Users/HP/Downloads/2017CS10387\_2017CS50417/Week2>
\item  Give Command For Compilation
\item cl Week2\_MKL.cpp  mkl\_intel\_lp64\_dll.lib mkl\_intel\_thread\_dll.lib
\item mkl\_core\_dll\.lib libiomp5md.lib\\
\item Week2\_MKL [\-\-commands\-\-]
\end{itemize}



\end{document}

