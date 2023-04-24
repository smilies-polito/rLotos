/*
PalaCell
Version: 0.2.0
Author: Raphael Conradin (raphael.conradin@unige.ch)
Creation: 22/05/2017
Copyright (c) Université de Genève
*/
#include <experimental/filesystem>
#include "palabos2D.h"
#include "palabos2D.hh"

#include "../src/palaCell2D.h"
#include "../src/palaCell2D.hh"
#include "../src/palaCellSimulation.cpp"

#define ADESCRIPTOR descriptors::AdvectionDiffusionD2Q5Descriptor
#define ADYNAMICS AdvectionDiffusionBGKdynamics

namespace fs = std::experimental::filesystem;

using namespace plb;
using namespace plb::descriptors;
using namespace plb::palacell;

typedef double T;

int main(int argc, char* argv[]) {
    /* The function can have two arguments:
          1) the parameters file (with the path from the executable)
          2) the output folder
    */
    plbInit(&argc, &argv);
    std::string paramFile = "simulation1.xml";
    std::string outputDir = "output/";
    if(argc >= 2) {
      paramFile = argv[1];
    }
    if(argc >= 3) {
      outputDir = argv[2];
    }
    if (!fs::is_directory(outputDir) || !fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }
    global::directories().setOutputDir(outputDir);
    global::timer("main").start();

    // Loading parameters
    Parametres<T> parameters(paramFile, ADESCRIPTOR<T>::cs2);
    WriteCSV2D csv(parameters.finalVTK, parameters.exportCSV, parameters.exportDBG);
    pcout << "Lattice dimensions: " << parameters.domain_lu.getNx()
                                    << ", " << parameters.domain_lu.getNy() << std::endl;
    pcout << "Seed for randomness in point position: " << parameters.seed << "\n";

    // Generate multiblocks
    MultiBlockLattice2D<T, ADESCRIPTOR>* speciesLattice
        = new MultiBlockLattice2D<T, ADESCRIPTOR> (
                        parameters.domain_lu.getNx(),parameters.domain_lu.getNy(),
                        new ADYNAMICS<T, ADESCRIPTOR>(1./parameters.tauDiffusive) );

    Group2D group(speciesLattice, "Species");
    group.generateParticleField<ParallelCellGeometry2D<T>::ParticleFieldT>("Particles", parameters.envelopeWidth);
    group.generateParticleField<ParallelCellGeometry2D<T>::ParticleFieldT>("WallParticles", parameters.envelopeWidth);
    group.generateScalar<plint>("Voxels", parameters.envelopeWidth);

    ParallelCellGeometry2D<T> parallelGeometry(group, parameters);
    SerialCellGeometry2D<T> serialGeometry(parameters);
    std::vector<SerialCellGeometry2D<T>::CellProperty> cellsProperties;

    ChemicalSpecies<T,ADESCRIPTOR> chemicalSpecies(group, parameters);

    Simulation<T,ADESCRIPTOR> simulation;

    // Geometry initialization
    // Cells
    if(!parameters.initialVTK.get().empty())
      serialGeometry.readCellVTK(parameters.initialVTK);
    if(serialGeometry.getNumCells() == 0) {
      pcout << "!***********\n!\n!   Warnings there is no cells in the initialization file.\n!";
      if(!(parameters.ix0==0 && parameters.iy0==0)){
        pcout << "   Adding cell in " << parameters.ix0 << " " << parameters.iy0 << ".\n!\n!***********\n";
        serialGeometry.generateCircleCellAt(parameters.toPhysicalUnits(Array<T,2>{parameters.ix0,parameters.iy0}), 10*std::sqrt(3/M_PI));
      }
      else {
        pcout << "   Adding cell in center of domain.\n!\n!***********\n";
        serialGeometry.generateCircleCellAt(parameters.toPhysicalUnits(Array<T,2>{parameters.domain_lu.getNx()/2.0,parameters.domain_lu.getNy()/2.0}), 10*std::sqrt(3/M_PI));
      }
    }
    serialGeometry.insertVerticesInCells();
    serialGeometry.removeVerticesInCells();

    // Walls
    if(!parameters.initialWallVTK.get().empty())
      serialGeometry.readWallVTK(parameters.initialWallVTK);
    // serialGeometry.generateCircleWallBoundary();
    if(serialGeometry.getNumWalls() > 0){
      serialGeometry.insertVerticesInWalls();
      serialGeometry.removeVerticesInWalls();
      serialGeometry.writeVTKWall(parameters.finalVTK.get(), group);
    }

    parallelGeometry.parallelize(serialGeometry);
    parallelGeometry.parallelSynchronizeSerialGeometry(serialGeometry);
    setToConstant<plint>(group.getScalar<plint>("Voxels"), group.getBoundingBox(), -1);
    parallelGeometry.computeNormals();
    parallelGeometry.voxelizeInnerBorders();
    parallelGeometry.voxelizeFull(serialGeometry);
    parallelGeometry.voxelizeOuterBorders();

    // Signaling initialization
    if(parameters.type < 0) {
      chemicalSpecies.initialise();
      chemicalSpecies.updateCellsSignal(serialGeometry.getNumCells());
    }

    // Simulation beginning
    for(plint i = parameters.initStep; parameters.testLoop(i,serialGeometry.getNumCells()); i++){
        std::flush(pcout.getOriginalStream());
        if(parameters.verbose > 0) {
          //pcout << "!!!!!!!! " << global::mpi().getRank() << " Iteration:" << i << "\n";
        }
        if(i == parameters.stopAt){
          parameters.comprForce = 0.0;
        }

        // Dynamique - Parallel
        // Compute dynamics
        parallelGeometry.advanceVertices();
        parallelGeometry.computeNormals();
        parallelGeometry.recomputeRedLinks();
        parallelGeometry.computeCellProperties<ADESCRIPTOR>(serialGeometry, chemicalSpecies);
        parallelGeometry.computeMassEvolution(serialGeometry);
        parallelGeometry.updatePressure(serialGeometry);
        parallelGeometry.computeFakeWall();
        parallelGeometry.computeVertexForces(serialGeometry);
        parallelGeometry.computeVertexVelocities(); // Calcul des forces, et de la vitesse résultante

        // Update geometry
        parallelGeometry.computeNormals();
        parallelGeometry.updateNPPos();
        bool change = false;
        parallelGeometry.removePoints(change, serialGeometry);
        parallelGeometry.checkIntegrity("remove");
        parallelGeometry.addPoints(change, serialGeometry);
        parallelGeometry.checkIntegrity("add");

        // Update grid
        parallelGeometry.voxelizeInnerBorders();
        parallelGeometry.voxelizeOuterBorders();

        // Update chemical
        if(parameters.type < 0) {
          chemicalSpecies.iterate();
          chemicalSpecies.updateCellsSignal(serialGeometry.getNumCells());
        }

        // Cell division and export
        parallelGeometry.updateGlob(serialGeometry, i);
        plint whichCell = -1;
        if((parallelGeometry.needsCellDivision(whichCell, serialGeometry)) | (i % parameters.exportStep == 0)) {
           // Serialisation
            if(change){
                parallelGeometry.recomputeRedLinks();
            }
            parallelGeometry.serialize(serialGeometry);
            parallelGeometry.parallelSynchronizeSerialGeometry(serialGeometry);

            // Export - Serial
            if(i % parameters.exportStep == 0){
               parallelGeometry.computeCellProperties<ADESCRIPTOR>(serialGeometry, chemicalSpecies);
               // parallelGeometry.computePointProperties(serialGeometry);
                serialGeometry.writeVTK(parameters.finalVTK.get()+"_" + std::to_string(i), group);
                if(parameters.exportSpecies && parameters.type < 0)
                  chemicalSpecies.writeVTK(parameters.finalVTK.get()+"_" + std::to_string(i));
                if(parameters.exportCSV)
                  serialGeometry.writeCSV(csv, i);
            }

            // Geometrie - Serial
            if(whichCell != -1) {
                serialGeometry.splitCell(whichCell);
                serialGeometry.insertVerticesInCells();
                serialGeometry.removeVerticesInCells();
                if(parameters.verbose > 0) {
                    pcout << "Numcells: " << serialGeometry.getNumCells() << " Iteration: " << i << "\n";
                }

                // Parallelization
                parallelGeometry.parallelize(serialGeometry);
                parallelGeometry.parallelSynchronizeSerialGeometry(serialGeometry);
                parallelGeometry.checkIntegrity("parallellize");
                parallelGeometry.computeNormals();
                setToConstant<plint>(group.getScalar<plint>("Voxels"), group.getBoundingBox(), -1);
                parallelGeometry.voxelizeInnerBorders();
                parallelGeometry.voxelizeFull(serialGeometry);
                parallelGeometry.voxelizeOuterBorders();
            }
        }
    }

    // Export final step
    parameters.exportCells = true;
    parallelGeometry.recomputeRedLinks();
    parallelGeometry.serialize(serialGeometry);
    parallelGeometry.parallelSynchronizeSerialGeometry(serialGeometry);
    parallelGeometry.computeCellProperties<ADESCRIPTOR>(serialGeometry, chemicalSpecies);
    serialGeometry.writeVTK(parameters.finalVTK.get()+"_final", group);

    // Test copy to a new serialGeometry
    SerialCellGeometry2D<T> newSerialGeometry(parameters);
    parallelGeometry.serialize(newSerialGeometry);
    parallelGeometry.parallelSynchronizeSerialGeometry(newSerialGeometry);
    parallelGeometry.computeCellProperties<ADESCRIPTOR>(newSerialGeometry, chemicalSpecies);
}
