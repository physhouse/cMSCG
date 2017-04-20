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
    normalVector.resize(info->numSplines);
    splineCoefficients.resize(info->numSplines);
    normalMatrix.resize(info->numSplines);
    printf("set up containers\n");
    for (auto&& i : normalMatrix)
    {
        i.resize(info->numSplines);
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
    
    double dx = trajFrame->positions[2*i][0] - trajFrame->positions[2*j][0];
    if (dx > 0.5 * info->boxLength) dx = dx - info->boxLength;
    if (dx < -0.5 * info->boxLength) dx = dx + info->boxLength;
    double dy = trajFrame->positions[2*i][1] - trajFrame->positions[2*j][1];
    if (dy > 0.5 * info->boxLength) dy = dy - info->boxLength;
    if (dy < -0.5 * info->boxLength) dy = dy + info->boxLength;
    double dz = trajFrame->positions[2*i][2] - trajFrame->positions[2*j][2];
    if (dz > 0.5 * info->boxLength) dz = dz - info->boxLength;
    if (dz < -0.5 * info->boxLength) dz = dz + info->boxLength;
    double dx1 = trajFrame->positions[2*i+1][0] - trajFrame->positions[2*j+1][0];
    if (dx1 > 0.5 * info->boxLength) dx1 = dx1 - info->boxLength;
    if (dx1 < -0.5 * info->boxLength) dx1 = dx1 + info->boxLength;
    double dy1 = trajFrame->positions[2*i+1][1] - trajFrame->positions[2*j+1][1];
    if (dy1 > 0.5 * info->boxLength) dy1 = dy1 - info->boxLength;
    if (dy1 < -0.5 * info->boxLength) dy1 = dy1 + info->boxLength;
    double dz1 = trajFrame->positions[2*i+1][2] - trajFrame->positions[2*j+1][2];
    if (dz1 > 0.5 * info->boxLength) dz1 = dz1 - info->boxLength;
    if (dz1 < -0.5 * info->boxLength) dz1 = dz1 + info->boxLength;

    double denominator = sqrt(mc*mc*(dx*dx+dy*dy+dz*dz) + mo*mo*(dx1*dx1+dy1*dy1+dz1*dz1) + 2.0*mo*mc*(dx*dx1+dy*dy1+dz*dz1));
    std::vector<double> result(3);
    result[0] = (mc*dx + mo*dx1) / denominator;
    result[1] = (mc*dy + mo*dy1) / denominator;
    result[2] = (mc*dz + mo*dz1) / denominator;
    //printf("dx %lf dx1 %lf dy %lf dy1 %lf dz %lf dz1 %lf result %lf %lf %lf\n", dx,dx1,dy,dy1,dz,dz1,result[0],result[1],result[2]);
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
    std::vector<std::vector<double> > frameMatrix(nSplines, std::vector<double>(3*nall));
    double normalFactor = 1.0 / info->steps;
   
    // Computing Matrix F_km 
    for (int i=0; i<nall; i++)
    {
        for (int j = i+1; j<nall; j++)
        {
            double riojc = distance(trajFrame->positions[2*i+1],trajFrame->positions[2*j]);
            double ricjo = distance(trajFrame->positions[2*i],trajFrame->positions[2*j+1]);
            double rjojc = distance(trajFrame->positions[2*j],trajFrame->positions[2*j+1]);
            double max = std::max(riojc, std::max(ricjo, rjojc));
            if (max < info->end)
            {
                //printf("i == %d j == %d\n", i, j);
                std::vector<double> eiojo = parallelUnitVector(2*i+1, 2*j+1);
                std::vector<double> eicjc = parallelUnitVector(2*i, 2*j);
                std::vector<double> ejojc = parallelUnitVector(2*j+1, 2*j);
                std::vector<double> ecom = centerOfMassUnitVector(i,j);
                //printf("ecom[0] = %lf\n", ecom[0]);
                
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
                
                auto invA = inverseMatrix(A); //TODO
                double alpha = invA[0][0]*b[0] + invA[0][1]*b[1] + invA[0][2]*b[2];
                double beta  = invA[1][0]*b[0] + invA[1][1]*b[1] + invA[1][2]*b[2];
                double gamma = invA[2][0]*b[0] + invA[2][1]*b[1] + invA[2][2]*b[2];
                double supergamma = gamma*100000000.0;
                printf("alpha = %lf, beta= %lf, gamma=%e supergamma=%e\n", alpha, beta, gamma, supergamma);
                // alpha, beta, gamma == b
                double kco = vectorSize(2*i,2*j,2*j,2*j+1);
                double koc = vectorSize(2*i+1,2*j+1,2*j+1,2*j);

                //printf("riojc = %lf ricjo = %lf rjojc = %lf\n", riojc,ricjo,rjojc);

                gsl_bspline_eval(riojc, splineValue, bw);
                //printf("passing\n");
                //printf("ecom[0] = %lf\n", ecom[0]);
            
                for (int m=0; m<nSplines; m++)
                {
                     double phim = gsl_vector_get(splineValue, m);
                     //printf("    loop m : %d phim = %lf\n",m, phim);
                     if (phim < 1e-20)
		         continue;
                     // For all three dimensions
                     //printf("hello world\n");
                     //printf("ecom[0] = %lf\n", ecom[0]);
                     //printf("frameMatrix[m][i] = %lf\n", frameMatrix[m][3*i]);
                     frameMatrix[m][3*i]     += phim * rjojc * ecom[0] / koc / gamma;
                     frameMatrix[m][3*i + 1] += phim * rjojc * ecom[1] / koc / gamma;
                     frameMatrix[m][3*i + 2] += phim * rjojc * ecom[2] / koc / gamma;
                     frameMatrix[m][3*j]     -= phim * rjojc * ecom[0] / koc / gamma;
                     frameMatrix[m][3*j + 1] -= phim * rjojc * ecom[1] / koc / gamma;
                     frameMatrix[m][3*j + 2] -= phim * rjojc * ecom[2] / koc / gamma;
                     //printf("    loop m : %d\n",m);
                }
                //printf("here\n");
                gsl_bspline_eval(ricjo, splineValue, bw);
                for (int m=0; m<nSplines; m++)
                {
                     double phim = gsl_vector_get(splineValue, m);
                     if (phim < 1e-20)
		         continue;
                     // For all three dimensions
                     frameMatrix[m][3*i]     -= phim * rjojc * ecom[0] / kco / gamma;
                     frameMatrix[m][3*i + 1] -= phim * rjojc * ecom[1] / kco / gamma;
                     frameMatrix[m][3*i + 2] -= phim * rjojc * ecom[2] / kco / gamma;
                     frameMatrix[m][3*j]     += phim * rjojc * ecom[0] / kco / gamma;
                     frameMatrix[m][3*j + 1] += phim * rjojc * ecom[1] / kco / gamma;
                     frameMatrix[m][3*j + 2] += phim * rjojc * ecom[2] / kco / gamma;
                }
                //printf("here2\n");
            }  
            //printf("endofLoop i == %d j == %d\n", i, j);
        
        }
    }

        
    // Constructing the normal Matrix and normal Vector
    for (int m=0; m<nSplines; m++)
    {
        for (int n=0; n<nSplines; n++)
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
    // Solving the least square normal equation G*phi = b
    double* G = new double[info->numSplines * info->numSplines];
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
    delete[] b;
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

    FILE* fp = fopen("gamma_out.dat", "w");

    while (start < end)
    {
        double gamma_r = 0.0;
        gsl_bspline_eval(start, splineValue, bw);
        for (int m=0; m<info->numSplines; m++)
        {
           gamma_r += splineCoefficients[m] * gsl_vector_get(splineValue, m);
        }
        fprintf(fp, "%lf\t%lf\n", start, gamma_r);
        start = start + precision;
    }
    
    fclose(fp);
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
