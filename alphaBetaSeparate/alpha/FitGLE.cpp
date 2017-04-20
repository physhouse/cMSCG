#include "FitGLE.h"
#include "comm.h"
#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>
#include <cassert>
#include <gsl/gsl_bspline.h>
#include <lapacke.h>

using namespace FITGLE_NS;

FitGLE::FitGLE(int argc, char** argv)
{
    if (argc != 3)
    {
        printf("./FitGLE.x [trajectory Filename] [number of Particles]\n");
    }

    assert(argc == 3);
    printf("Initializing FitGLE parameters...\n");

    // parsing the configuration parameters
    info = std::make_shared<InputParameters>();
    VAR_BEGIN
      GET_REAL(info->start)
      GET_REAL(info->end)
      GET_REAL(info->boxLength)
      GET_REAL(info->outputPrecision)
      GET_INT(info->splineOrder)
      GET_INT(info->numSplines)
      GET_INT(info->steps)
    VAR_END
           
    printf("set up trajectory files\n");
    trajFrame = std::make_shared<Frame>(atoi(argv[2]), argv[1]);
    // Initialize the Normal Equation matrix and vectors
    // Set up the size of splines according to order and numbers
    printf("set up b-spline data structures\n");
    int numBreaks = info->numSplines + 2 - info->splineOrder;
    normalVector.resize(2*info->numSplines);
    splineCoefficients.resize(2*info->numSplines);
    normalMatrix.resize(2*info->numSplines);
    printf("set up containers\n");
    for (auto&& i : normalMatrix)
    {
        i.resize(2*info->numSplines);
    }

    // Initialize the spline set up
    bw = gsl_bspline_alloc(info->splineOrder, numBreaks);
    splineValue = gsl_vector_alloc(info->numSplines);
    gsl_bspline_knots_uniform(info->start, info->end, bw);
    printf("finishing configuration, entering normal equation accumulation\n");
}

FitGLE::~FitGLE()
{
    gsl_bspline_free(bw);
    gsl_vector_free(splineValue);
    printf("Exiting the Fitting GLE process...\n");
}

inline double FitGLE::distance(std::vector<double> & A, std::vector<double> & B)
{
    double dx = A[0] - B[0];
    if (dx > 0.5 * info->boxLength) dx = dx - info->boxLength;
    if (dx < -0.5 * info->boxLength) dx = dx + info->boxLength;
    double dy = A[1] - B[1];
    if (dy > 0.5 * info->boxLength) dy = dy - info->boxLength;
    if (dy < -0.5 * info->boxLength) dy = dy + info->boxLength;
    double dz = A[2] - B[2];
    if (dz > 0.5 * info->boxLength) dz = dz - info->boxLength;
    if (dz < -0.5 * info->boxLength) dz = dz + info->boxLength;
  
    return sqrt(dx*dx + dy*dy + dz*dz);
}

inline std::vector<double> FitGLE::parallelVelocity(int i, int j)
{
    double dx = trajFrame->positions[i][0] - trajFrame->positions[j][0];
    if (dx > 0.5 * info->boxLength) dx = dx - info->boxLength;
    if (dx < -0.5 * info->boxLength) dx = dx + info->boxLength;
    double dy = trajFrame->positions[i][1] - trajFrame->positions[j][1];
    if (dy > 0.5 * info->boxLength) dy = dy - info->boxLength;
    if (dy < -0.5 * info->boxLength) dy = dy + info->boxLength;
    double dz = trajFrame->positions[i][2] - trajFrame->positions[j][2];
    if (dz > 0.5 * info->boxLength) dz = dz - info->boxLength;
    if (dz < -0.5 * info->boxLength) dz = dz + info->boxLength;

    double rij = sqrt(dx*dx + dy*dy + dz*dz);
    double eij[] = {dx/rij, dy/rij, dz/rij};
    std::vector<double> vij;
    std::transform(trajFrame->velocities[i].begin(), trajFrame->velocities[i].end(), trajFrame->velocities[j].begin(), std::back_inserter(vij), std::minus<double>());
     
    double projection = vij[0] * eij[0] + vij[1] * eij[1] + vij[2] * eij[2];
    vij[0] = projection * eij[0];
    vij[1] = projection * eij[1];
    vij[2] = projection * eij[2];
    return vij;
}

inline std::vector<double> FitGLE::parallelUnitVector(int i, int j)
{
    double dx = trajFrame->positions[i][0] - trajFrame->positions[j][0];
    if (dx > 0.5 * info->boxLength) dx = dx - info->boxLength;
    if (dx < -0.5 * info->boxLength) dx = dx + info->boxLength;
    double dy = trajFrame->positions[i][1] - trajFrame->positions[j][1];
    if (dy > 0.5 * info->boxLength) dy = dy - info->boxLength;
    if (dy < -0.5 * info->boxLength) dy = dy + info->boxLength;
    double dz = trajFrame->positions[i][2] - trajFrame->positions[j][2];
    if (dz > 0.5 * info->boxLength) dz = dz - info->boxLength;
    if (dz < -0.5 * info->boxLength) dz = dz + info->boxLength;

    double rij = sqrt(dx*dx + dy*dy + dz*dz);
    std::vector<double> eij;
    eij.push_back(dx / rij);
    eij.push_back(dy / rij);
    eij.push_back(dz / rij);
    return eij;
}

inline std::vector<double> FitGLE::centerOfMassUnitVector(int i, int j)
{
    double mc = 15.035;
    double mo = 17.007;

    std::vector<double> ovectori(trajFrame->positions[2*i+1]);
    std::vector<double> cvectori(trajFrame->positions[2*i]);
    std::vector<double> delta;
    std::transform(ovectori.begin(), ovectori.end(), cvectori.begin(), std::back_inserter(delta), std::minus<double>());
    for (auto && item : delta)
    {
        if (item > 0.5 * info->boxLength) item -= info->boxLength;
        else if (item < -0.5 * info->boxLength) item += info->boxLength;
    }
    std::transform(delta.begin(), delta.end(), cvectori.begin(), ovectori.begin(), std::plus<double>());
    for (int i=0; i<3; i++)
    {
        cvectori[i] = (mc * cvectori[i] + mo * ovectori[i]) / (mc + mo);
        if (cvectori[i] > info->boxLength) cvectori[i] -= info->boxLength;
        else if (cvectori[i] < 0.0) cvectori[i] += info->boxLength;
    }

    std::vector<double> ovectorj(trajFrame->positions[2*j+1]);
    std::vector<double> cvectorj(trajFrame->positions[2*j]);
    std::transform(ovectorj.begin(), ovectorj.end(), cvectorj.begin(), delta.begin(), std::minus<double>());
    for (auto && item : delta)
    {
        if (item > 0.5 * info->boxLength) item -= info->boxLength;
        else if (item < -0.5 * info->boxLength) item += info->boxLength;
    }
    std::transform(delta.begin(), delta.end(), cvectorj.begin(), ovectorj.begin(), std::plus<double>());
    for (int i=0; i<3; i++)
    {
        cvectorj[i] = (mc * cvectorj[i] + mo * ovectorj[i]) / (mc + mo);
        if (cvectorj[i] > info->boxLength) cvectorj[i] -= info->boxLength;
        else if (cvectorj[i] < 0.0) cvectorj[i] += info->boxLength;
    }

    std::vector<double> result(3);
    double distance = 0.0;
    for (int i=0; i<3; i++)
    {
       double dx = cvectori[i] - cvectorj[i];
       if (dx > 0.5*info->boxLength) dx -= info->boxLength;
       else if (dx < -0.5*info->boxLength) dx += info->boxLength;
       result[i] = dx;
       distance += dx * dx;
    }
    distance = sqrt(distance);
    for (int i=0; i<3; i++) result[i] = result[i] / distance;
    return result;
}

inline double FitGLE::vectorSize(int i1, int j1, int i2, int j2)
{
    double dx = trajFrame->positions[i1][0] - trajFrame->positions[j1][0];
    if (dx > 0.5 * info->boxLength) dx = dx - info->boxLength;
    if (dx < -0.5 * info->boxLength) dx = dx + info->boxLength;
    double dy = trajFrame->positions[i1][1] - trajFrame->positions[j1][1];
    if (dy > 0.5 * info->boxLength) dy = dy - info->boxLength;
    if (dy < -0.5 * info->boxLength) dy = dy + info->boxLength;
    double dz = trajFrame->positions[i1][2] - trajFrame->positions[j1][2];
    if (dz > 0.5 * info->boxLength) dz = dz - info->boxLength;
    if (dz < -0.5 * info->boxLength) dz = dz + info->boxLength;
    double dx1 = trajFrame->positions[i2][0] - trajFrame->positions[j2][0];
    if (dx1 > 0.5 * info->boxLength) dx1 = dx1 - info->boxLength;
    if (dx1 < -0.5 * info->boxLength) dx1 = dx1 + info->boxLength;
    double dy1 = trajFrame->positions[i2][1] - trajFrame->positions[j2][1];
    if (dy1 > 0.5 * info->boxLength) dy1 = dy1 - info->boxLength;
    if (dy1 < -0.5 * info->boxLength) dy1 = dy1 + info->boxLength;
    double dz1 = trajFrame->positions[i2][2] - trajFrame->positions[j2][2];
    if (dz1 > 0.5 * info->boxLength) dz1 = dz1 - info->boxLength;
    if (dz1 < -0.5 * info->boxLength) dz1 = dz1 + info->boxLength;
    double denominator = sqrt((dx*dx+dy*dy+dz*dz) + (dx1*dx1+dy1*dy1+dz1*dz1) + 2.0*(dx*dx1+dy*dy1+dz*dz1));
    return denominator;
}

std::vector<std::vector<double> >  FitGLE::inverseMatrix(std::vector<std::vector<double> > A)
{
    double a = A[0][0];
    double b = A[0][1];
    double c = A[0][2];
    double d = A[1][0];
    double e = A[1][1];
    double f = A[1][2];
    double g = A[2][0];
    double h = A[2][1];
    double i = A[2][2];
    double det = a*e*i - a*f*h + b*f*g - b*d*i + c*d*h - c*e*g;
    std::vector<std::vector<double> > result(3, std::vector<double>(3));
    result[0][0] = (e*i - f*h) / det;
    result[0][1] = (c*h - b*i) / det;
    result[0][2] = (b*f - c*e) / det;
    result[1][0] = (f*g - d*i) / det;
    result[1][1] = (a*i - c*g) / det;
    result[1][2] = (c*d - a*f) / det;
    result[2][0] = (d*h - e*g) / det;
    result[2][1] = (b*g - a*h) / det;
    result[2][2] = (a*e - b*d) / det;
    return result;
}

// Accumulate the normal equation for this particular frame
void FitGLE::accumulateNormalEquation()
{
    int nall = trajFrame->numParticles / 2;
    int nSplines = info->numSplines;
    std::vector<std::vector<double> > frameMatrix(2*nSplines, std::vector<double>(3*nall));
    double normalFactor = 1.0 / info->steps;
   
    // Computing Matrix F_km 
    for (int i=0; i<nall; i++)
    {
        for (int j = i+1; j<nall; j++)
        {
            double riojc = distance(trajFrame->positions[2*i+1],trajFrame->positions[2*j]);
            double riojo = distance(trajFrame->positions[2*i+1],trajFrame->positions[2*j+1]);
            double ricjo = distance(trajFrame->positions[2*i],trajFrame->positions[2*j+1]);
            double rjojc = distance(trajFrame->positions[2*j],trajFrame->positions[2*j+1]);
            double max = std::max(std::max(riojo, riojc), std::max(ricjo, rjojc));
            if (max < info->end && max > info->start)
            {
                std::vector<double> eiojo = parallelUnitVector(2*i+1, 2*j+1);
                std::vector<double> eicjc = parallelUnitVector(2*i, 2*j);
                std::vector<double> ejojc = parallelUnitVector(2*j+1, 2*j);
                std::vector<double> ecom = centerOfMassUnitVector(i,j);
                
                std::vector<std::vector<double> > A(3, std::vector<double>(3));
                A[0][0] = eiojo[0];
                A[0][1] = eicjc[0];
                A[0][2] = ejojc[0];
                A[1][0] = eiojo[1];
                A[1][1] = eicjc[1];
                A[1][2] = ejojc[1];
                A[2][0] = eiojo[2];
                A[2][1] = eicjc[2];
                A[2][2] = ejojc[2];
 
                std::vector<double> b(3);
                b[0] = ecom[0];
                b[1] = ecom[1];
                b[2] = ecom[2];
                
                auto invA = inverseMatrix(A);
                double alpha = invA[0][0]*b[0] + invA[0][1]*b[1] + invA[0][2]*b[2];
                double beta  = invA[1][0]*b[0] + invA[1][1]*b[1] + invA[1][2]*b[2];
                double gamma = invA[2][0]*b[0] + invA[2][1]*b[1] + invA[2][2]*b[2];
                // alpha, beta, gamma == b
                double kco = vectorSize(2*i,2*j,2*j,2*j+1);
                double koc = vectorSize(2*i+1,2*j+1,2*j+1,2*j);

                gsl_bspline_eval(riojo, splineValue, bw);
                for (int m=0; m<nSplines; m++)
                {
                     double phim = gsl_vector_get(splineValue, m);
                     if (phim < 1e-20)
		         continue;
                     // For all three dimensions
                     frameMatrix[m][3*i]     += phim * ecom[0] / alpha;
                     frameMatrix[m][3*i + 1] += phim * ecom[1] / alpha;
                     frameMatrix[m][3*i + 2] += phim * ecom[2] / alpha;
                     frameMatrix[m][3*j]     -= phim * ecom[0] / alpha;
                     frameMatrix[m][3*j + 1] -= phim * ecom[1] / alpha;
                     frameMatrix[m][3*j + 2] -= phim * ecom[2] / alpha;
                }

                gsl_bspline_eval(riojc, splineValue, bw);
                for (int m=0; m<nSplines; m++)
                {
                     double phim = gsl_vector_get(splineValue, m);
                     if (phim < 1e-20)
		         continue;
                     // For all three dimensions
                     frameMatrix[nSplines + m][3*i]     += phim * riojo * ecom[0] / koc / alpha;
                     frameMatrix[nSplines + m][3*i + 1] += phim * riojo * ecom[1] / koc / alpha;
                     frameMatrix[nSplines + m][3*i + 2] += phim * riojo * ecom[2] / koc / alpha;
                     frameMatrix[nSplines + m][3*j]     -= phim * riojo * ecom[0] / koc / alpha;
                     frameMatrix[nSplines + m][3*j + 1] -= phim * riojo * ecom[1] / koc / alpha;
                     frameMatrix[nSplines + m][3*j + 2] -= phim * riojo * ecom[2] / koc / alpha;
                }
            }  
        }
    }

        
    // Constructing the normal Matrix and normal Vector
    for (int m=0; m<2*nSplines; m++)
    {
        for (int n=0; n<2*nSplines; n++)
        {
            double sum = 0.0;
            for (int k=0; k<3 * nall; k++)
                sum += frameMatrix[m][k] * frameMatrix[n][k];
            normalMatrix[m][n] += sum * normalFactor;
        }
   
        double sum_b = 0.0; 
        for (int k=0; k<3 * nall; k++)
            sum_b += frameMatrix[m][k] * (trajFrame->residualForces[2*(k/3)][k%3] + trajFrame->residualForces[2*(k/3)+1][k%3]);
        normalVector[m] += sum_b * normalFactor;
    }
}

void FitGLE::leastSquareSolver()
{
    int basisSize = 2*info->numSplines;
    double* G = new double[basisSize * basisSize];
    double* b = new double[basisSize];


    std::vector<double> h(basisSize, 0.0);
    for (int i = 0; i < basisSize; i++) {
        for (int j = 0; j < basisSize; j++) {            
            h[j] = h[j] + normalMatrix[i][j] * normalMatrix[i][j];  
        }
    }
    for (int i = 0; i < basisSize; i++) {
        if (h[i] < 1.0E-20) h[i] = 1.0;
        else h[i] = 1.0 / sqrt(h[i]);
    }
    for (int i =0; i < basisSize; i++)
    {
        for (int j=0; j < basisSize; j++)
           normalMatrix[i][j] *= h[j];
    }

    for (int m=0; m<basisSize; m++)
    {
        for (int n=0; n<basisSize; n++)
        {
            G[m*basisSize + n] = normalMatrix[m][n];
        }
        b[m] = normalVector[m];
        printf("m %d %lf\n", m, b[m]);
    }
    int m = basisSize;
    int n = basisSize;
    int nrhs = 1;
    int lda = basisSize;
    int ldb = 1;
    double rcond = -1.0;
    int irank;
    double* singularValue = new double[basisSize];
    int solverInfo = LAPACKE_dgelss(LAPACK_ROW_MAJOR, m, n, nrhs, G, lda, b, ldb, singularValue, rcond, &irank);

    printf("LSQ Solver Info: %d\n", solverInfo);

    for (int m=0; m<basisSize; m++)
    {
        splineCoefficients[m] = b[m] * h[m];
        printf("spline : %lf\n", splineCoefficients[m]);
    }

    delete[] G;
    delete[] b;
    // Solving the least square normal equation G*phi = b
    /*double* G = new double[info->numSplines * info->numSplines];
    double* b = new double[info->numSplines];

    
    // Preconditioning the Normal Matrix
    std::vector<double> h(info->numSplines, 0.0);
    for (int i = 0; i < info->numSplines; i++) {
        for (int j = 0; j < info->numSplines; j++) {
            h[j] = h[j] + normalMatrix[i][j] * normalMatrix[i][j];  //mat->dense_fm_matrix[j * mat->accumulation_matrix_rows + i] * mat->dense_fm_matrix[j * mat->accumulation_matrix_rows + i];
        }
    }
    for (int i = 0; i < info->numSplines; i++) {
        if (h[i] < 1.0E-20) h[i] = 1.0;
        else h[i] = 1.0 / sqrt(h[i]);
    }
    for (int i =0; i < info->numSplines; i++)
    {
        for (int j=0; j < info->numSplines; j++)
           normalMatrix[i][j] *= h[j];
    }


    // Store the normalMatrix in container 
    for (int m=0; m<info->numSplines; m++)
    {
        for (int n=0; n<info->numSplines; n++)
        {
            G[m*info->numSplines + n] = normalMatrix[m][n];
        }
        b[m] = normalVector[m];
        printf("m %d %lf\n", m, b[m]);
    }
    

    // Solving the linear system using SVD

    int m = info->numSplines;
    int n = info->numSplines;
    int nrhs = 1;
    int lda = info->numSplines;
    int ldb = 1;
    double rcond = -1.0;
    int irank;
    double* singularValue = new double[info->numSplines];
    int solverInfo = LAPACKE_dgelss(LAPACK_ROW_MAJOR, m, n, nrhs, G, lda, b, ldb, singularValue, rcond, &irank); 
   
    printf("LSQ Solver Info: %d\n", solverInfo);

    for (int m=0; m<info->numSplines; m++)
        splineCoefficients[m] = b[m] * h[m];

    delete[] G;
    delete[] b;*/
}

// Output function for gamma(R)
void FitGLE::output()
{
    printf("output\n");
    double start = info->start;
    double end = info->end;
    double precision = info->outputPrecision;

    FILE* fb = fopen("spline_coeff.dat", "w");
    for (int m=0; m<info->numSplines; m++)
    {
        fprintf(fb, "%lf\n", splineCoefficients[m]);
    }
    fclose(fb);

    FILE* fp = fopen("foo_out.dat", "w");
    FILE* fp1= fopen("foc_out.dat", "w");

    while (start < end)
    {
        double gamma_r = 0.0;
        double foc_r = 0.0;
        gsl_bspline_eval(start, splineValue, bw);
        for (int m=0; m<info->numSplines; m++)
        {
           gamma_r += splineCoefficients[m] * gsl_vector_get(splineValue, m);
           foc_r   += splineCoefficients[m+info->numSplines] * gsl_vector_get(splineValue, m);
        }
        fprintf(fp, "%lf\t%lf\n", start, gamma_r);
        fprintf(fp1, "%lf\t%lf\n", start, foc_r);
        start = start + precision;
    }
    
    fclose(fp);
    fclose(fp1);
}

// Execution Process
void FitGLE::exec()
{
    printf("Accumulating the LSQ normal Matrix\n");
    for (int i=0; i<info->steps; i++)
    {
        trajFrame->readFrame();
        accumulateNormalEquation();
        printf("finishing step %d (total %d)\r", i+1, info->steps);
    }
    printf("\n");
    leastSquareSolver();
    output();
}
