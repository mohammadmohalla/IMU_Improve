#ifndef MATRIX_H_
#define MATRIX_H_

#include <opencv2/core/core.hpp>
#include "structs.h"

typedef struct
{
    int numRows;
    int numCols;
    double *data;
} Matrix;

//Set a value in the matrix
void set(Matrix *m, int row, int column, double value);

//Insert one Matrix into another
void insert(Matrix *target, Matrix *source, destY, destX);

//Add two matrices
void add(Matrix *a, Matrix *b, Matrix *c);

//Subtract one Matrix from the other
void subtract(Matrix *a, Matrix *b, Matrix *c);

//Subtract without the need for an additional matrix
void subtractEquals(Matrix *a, Matrix *b);

//Multiply one matrix with a value
void mult(Matrix *a, double value, Matrix *c);

//Multiply to matrices
void mult(Matrix *a, Matrix *b, Matrix *c);

//Multply and with the transposed of the second matrix
void multTransB(Matrix *a, Matrix *b, Matrix *c);

//Invert the matrix
bool invert(Matrix *a, Matrix *b);

#endif /* MATRIX_H_ */
