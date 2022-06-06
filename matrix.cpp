#include "matrix.h"

#include <math.h>
#include <vector>


#define log(...) __android_log_print(ANDROID_LOG_INFO, "SensorTest", __VA_ARGS__)

using namespace std;

//Set a value in the matrix
void set(Matrix *m, int row, int col, double value) {
    if( col < 0 || col >= m->numCols || row < 0 || row >= m->numRows ) {
        return;
    }

    m->data[ row * m->numCols + col ] = value; 
}

//Insert one Matrix into another
void insert(Matrix *target, Matrix *source) {
    for (int row = 0, row < source->numRows; row++) {
        int row_t = row * target->numCols;
        int row_s = row * source->numCols;
        for (int col = 0; col < source->numCols; col++) {
            target->data[row_t + col] = source->data[row_s + col];
        }
    }
}

//Add two matrices
void add(Matrix *a, Matrix *b, Matrix *c) {
    if( a->numCols != b->numCols || a->numRows != b->numRows || a->numCols != c->numCols || a->numRows != c->numRows ) {
        return;
    }

    int length = a->numRows*a->numCols;

    for(int i = 0; i < length; i++) {
        c->data[i] = a->data[i] + b->data[i];
    }
}

//Add two matrices
void add(Matrix *a, int row, int col, double value) {
    if( col < 0 || col >= m->numCols || row < 0 || row >= m->numRows ) {
        return;
    }

    m->data[ row * m->numCols + col ] += value; 
}

//Subtract one Matrix from the other
void subtract(Matrix *a, Matrix *b, Matrix *c) {
    if( a->numCols != b->numCols || a->numRows != b->numRows || a->numCols != c->numCols || a->numRows != c->numRows ) {
        return;
    }

    int length = a->numRows*a->numCols;

    for(int i = 0; i < length; i++) {
        c->data[i] = a->data[i] - b->data[i];
    }
}

//Subtract without the need for an additional matrix
void subtractEquals(Matrix *a, Matrix *b) {
    if( a->numCols != b->numCols || a->numRows != b->numRows) {
        return;
    }

    int length = a->numRows*a->numCols;

    for(int i = 0; i < length; i++) {
        a->data[i] -= b->data[i];
    }
}

//Multiply one matrix with a value
void mult(Matrix *a, double value) {
    int length = a->numRows*a->numCols;

    for(int i = 0; i < length; i++) {
        a->data[i] *= value;
    }
}

//Multiply to matrices
void mult(Matrix *a, Matrix *b, Matrix *c) {
    if( a->numCols != b->numRows ) {
        return;
    }
    else if( a->numRows != c->numRows || b->numCols != c->numCols ) {
        return;
    }

    int aIndexStart = 0;
    int cIndex = 0;

    for(int i = 0; i < a->numRows; i++) {
        for(int j = 0; j < b->numCols; j++) {
            double total = 0;

            int indexA = aIndexStart;
            int indexB = j;
            int end = indexA + b->numRows;
            
            while( indexA < end ) {
                total += a->data[indexA++] * b->data[indexB];
                indexB += b->numCols;
            }
            c->data[cIndex++] = total;
        }
        aIndexStart += a->numCols;
    }
}

//Multply and with the transposed of the second matrix
void multTransB(Matrix *a, Matrix *b, Matrix *c) {
    if( a->numCols != b->numRows ) {
        return;
    }
    else if( a->numRows != c->numRows || b->numCols != c->numCols ) {
        return;
    }

    int cIndex = 0;
    int aIndexStart = 0;

    for(int xA = 0; xA < a->numRows; xA++ ) {
        int end = aIndexStart + b->numCols;
        int indexB = 0;
        for( int xB = 0; xB < b->numRows; xB++ ) {
            int indexA = aIndexStart;
            double total = 0;

            while(indexA < end ) {
                total += a->data[indexA++] * b->[indexB++];
            }
            c->data[cIndex++] = total;
        }
        aIndexStart += a->numCols;
    }
}

//Invert the matrix
bool invert(Matrix *a, Matrix *b) {
    if( a->numRows != A->numCols ) {
        return false;
    }

    if (decompose(a)) {
        n = a->numCols;
        vv = decomposer._getVV();
        t = decomposer.getT().data;
        
        //invert
        if( b->numRows != n || b->numCols != n ) {
            return false;
        }

        double a[] = inv.data;

        for(int i =0; i < n; i++ ) {
            double el_ii = t[i*n+i];
            for( int j = 0; j <= i; j++ ) {
                double sum = (i==j) ? 1.0 : 0;
                for( int k=i-1; k >=j; k-- ) {
                    sum -= t[i*n+k]*a[j*n+k];
                }
                a[j*n+i] = sum / el_ii;
            }
        }
        // solve the system and handle the previous solution being in the upper triangle
        // takes advantage of symmetry
        for( int i=n-1; i>=0; i-- ) {
            double el_ii = t[i*n+i];

            for( int j = 0; j <= i; j++ ) {
                double sum = (i<j) ? 0 : a[j*n+i];
                for( int k=i+1;k<n;k++) {
                    sum -= t[k*n+i]*a[j*n+k];
                }
                a[i*n+j] = a[j*n+i] = sum / el_ii;
            }
        }
        
        return true;
    }
    else {
        return false;
    }  
}
