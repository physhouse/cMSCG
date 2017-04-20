#ifndef _FITGLE_H_
#define _FITGLE_H_

#include <cstdlib>
#include <cstdio>
#include <vector>
#include "Frame.h"
#include <memory>
#include <gsl/gsl_bspline.h>

namespace FITGLE_NS {

struct InputParameters
{
    double start;  //start distance r0
    double end;    //end distance r1
    int    splineOrder;
    int    numSplines;
    int    tableLength;
    double tableStart;
    double tableInterval;
    double outputPrecision;
    double boxLength; 
    int    steps;
    FILE*  fileTraj;
};  //Structure to store input parameters

class FitGLE
{
public:
    FitGLE(int argc, char** argv);
    ~FitGLE();
    void exec();

    //helper functions
    void accumulateNormalEquation();
    void leastSquareSolver();
    void output();
    
    //helper functions
    double distance(std::vector<double> &, std::vector<double> &);
    std::vector<double> parallelVelocity(int i, int j);
    std::vector<double> parallelUnitVector(int i, int j);
    std::vector<double> centerOfMassUnitVector(int i, int j);
    double centerOfMassDistance(int i, int j);
    double vectorSize(int i1, int j1, int i2, int j2);
    std::vector<std::vector<double> > inverseMatrix(std::vector<std::vector<double> >);

private:
    std::shared_ptr<class Frame> trajFrame;
    //class Frame* trajFrame;
    std::shared_ptr<struct InputParameters> info; 
    std::vector<double> divPoints;  // divide points of the b-spline radial ranges
    std::vector<std::vector<double> > normalMatrix;
    std::vector<double> normalVector;
    std::vector<double> splineCoefficients;
    std::vector<std::vector<double> > table;

    //gsl members for b-splines
    gsl_bspline_workspace *bw;
    gsl_vector *splineValue;

};

}

#endif
