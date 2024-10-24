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

#include "Generators/GeneratorHybrid.h"
#include <fairlogger/Logger.h>
#include <algorithm>

namespace o2
{
  namespace eventgen
  {
  GeneratorHybrid::GeneratorHybrid(std::vector<std::string> inputgens)
  {
    auto extname = GeneratorFromO2KineParam::Instance().fileName;
    for(auto gen : inputgens)
    {
      // Search if the generator name is inside generatorNames (which is a vector of strings)
      LOG(info) << "Checking if generator " << gen << " is in the list of available generators \n";
      if (std::find(generatorNames.begin(), generatorNames.end(), gen) != generatorNames.end())
      {
        LOG(info) << "Found generator " << gen << " in the list of available generators \n";
        if(gen.compare("boxgen") == 0){
            gens.push_back(std::make_unique<o2::eventgen::BoxGenerator>(22, 10, -5, 5, 0, 10, 0, 360));
            mGens.push_back(gen);
        } else if (gen.compare(0, 7, "pythia8") == 0) {
            gens.push_back(std::make_unique<o2::eventgen::GeneratorPythia8>());
            mGens.push_back(gen);
        } else if(gen.compare("extkinO2") == 0){
          gens.push_back(std::make_unique<o2::eventgen::GeneratorFromO2Kine>(extname.c_str()));
          mGens.push_back(gen);
        } else {
            LOG(info) << "Generator " << gen << " not found in the list of available generators \n";
        }
      }
    }  
  }

  Bool_t GeneratorHybrid::Init()
  {
    // init all sub-gens
    int count = 0;
    for (auto& gen : gens)
    {
      if (mGens[count] == "pythia8"){
        auto config = std::string();
        LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
        dynamic_cast<o2::eventgen::GeneratorPythia8*>(gen.get())->setConfig(config);
      } else if (mGens[count] == "pythia8pp"){
        auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_inel.cfg";
        LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
        dynamic_cast<o2::eventgen::GeneratorPythia8*>(gen.get())->setConfig(config);
      } else if (mGens[count] == "pythia8hf") {
        auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_hf.cfg";
        LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
        dynamic_cast<o2::eventgen::GeneratorPythia8*>(gen.get())->setConfig(config);
      } else if (mGens[count] == "pythia8hi") {
        auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_hi.cfg";
        LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
        dynamic_cast<o2::eventgen::GeneratorPythia8*>(gen.get())->setConfig(config);
      } else if (mGens[count] == "pythia8powheg") {
        auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_powheg.cfg";
        LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
        dynamic_cast<o2::eventgen::GeneratorPythia8*>(gen.get())->setConfig(config);
      }
      gen->Init();
      addSubGenerator(count, mGens[count]);
      count++;
    }
    return Generator::Init();
  }

  Bool_t GeneratorHybrid::generateEvent()
  {
      // here we call the individual gun generators in turn
      // (but we could easily call all of them to have cocktails)
      mIndex = gRandom->Integer(gens.size());
      LOG(info) << "GeneratorHybrid: generating event with generator " << mGens[mIndex];
      gens[mIndex]->clearParticles(); // clear container of this class
      gens[mIndex]->generateEvent();
      // notify the sub event generator
      notifySubGenerator(mIndex);
      return true;
  }      

  Bool_t GeneratorHybrid::importParticles()
  {
    mParticles.clear(); // clear container of mother class
    gens[mIndex]->importParticles();
    std::copy(gens[mIndex]->getParticles().begin(), gens[mIndex]->getParticles().end(), std::back_insert_iterator(mParticles));

    // we need to fix particles statuses --> need to enforce this on the importParticles level of individual generators
    for (auto& p : mParticles)
    {
      auto st = o2::mcgenstatus::MCGenStatusEncoding(p.GetStatusCode(), p.GetStatusCode()).fullEncoding;
      p.SetStatusCode(st);
      p.SetBit(ParticleStatus::kToBeDone, true);
    }

    return true;
  }     

  } // namespace eventgen
} // namespace o2

ClassImp(o2::eventgen::GeneratorHybrid);