// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author M. Giacalone - October 2024

#ifndef ALICEO2_EVENTGEN_GENERATORHYBRID_H_
#define ALICEO2_EVENTGEN_GENERATORHYBRID_H_

#include "Generators/Generator.h"
#include "Generators/BoxGenerator.h"
#include <Generators/GeneratorPythia8.h>
#include <Generators/GeneratorFromFile.h>
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCGenProperties.h"
#include "SimulationDataFormat/ParticleStatus.h"
#include "Generators/GeneratorHybridParam.h"
#include "Generators/GeneratorFromO2KineParam.h"
#include <Generators/GeneratorPythia8.h>
#include <TRandom3.h>

namespace o2
{
namespace eventgen
{

class GeneratorHybrid : public Generator
{

 public:
  GeneratorHybrid() = default;
  GeneratorHybrid(std::vector<std::string> gens);
  ~GeneratorHybrid() = default;
  
  Bool_t Init() override;
  Bool_t generateEvent() override;
  Bool_t importParticles() override;

 private:
  o2::eventgen::Generator* currentgen = nullptr;
  std::vector<std::unique_ptr<o2::eventgen::Generator>> gens;
  const std::vector<std::string> generatorNames = {"extkinO2", "boxgen", "external", "pythia8", "pythia8pp", "pythia8hi", "pythia8hf", "pythia8powheg"};
  std::vector<std::string> mGens;
  int mIndex = -1;
};

} // namespace eventgen
} // namespace o2

#endif
