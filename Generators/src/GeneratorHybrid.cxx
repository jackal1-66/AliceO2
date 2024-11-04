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
#include <FairPrimaryGenerator.h>

namespace o2
{
  namespace eventgen
  {
  GeneratorHybrid::GeneratorHybrid(const std::vector<std::string>& inputgens)
  {
    auto configs = GeneratorHybridParam::Instance().configs;
    mRandomize = GeneratorHybridParam::Instance().randomize;
    std::stringstream ss(configs);
    std::string conf;
    while (std::getline(ss, conf, ':')) {
      mConfigs.push_back(conf);
    }
    if(mConfigs.size() != inputgens.size()){
      LOG(fatal) << "Number of configurations does not match the number of generators";
    }
    if(mConfigs.size() == 0){
      for(auto gen : inputgens){
        mConfigs.push_back("");
      }
    }
    int index = 0;
    if (!mRandomize) {
      std::string fractions = GeneratorHybridParam::Instance().fractions;
      if(fractions.compare("") == 0){
        for (auto gen : inputgens) {
          mFractions.push_back(1);
        }
      }
      else{
        std::stringstream streamfrac(fractions);
        std::string frac;
        while (std::getline(streamfrac, frac, ',')) {
          if(frac.compare("") == 0)
            mFractions.push_back(1);
          else
            mFractions.push_back(std::stoi(frac));
        }
        if(mFractions.size() != inputgens.size()){
          LOG(fatal) << "Number of fractions does not match the number of generators";
          return;
        }
        // Check if all elements of mFractions are 0
        if (std::all_of(mFractions.begin(), mFractions.end(), [](int i){ return i == 0; })) {
          LOG(fatal) << "All fractions provided are 0, no simulation will be performed";
          return;
        }
        
      }  
    }
    for(auto gen : inputgens)
    {
      // Search if the generator name is inside generatorNames (which is a vector of strings)
      LOG(info) << "Checking if generator " << gen << " is in the list of available generators \n";
      if (std::find(generatorNames.begin(), generatorNames.end(), gen) != generatorNames.end())
      {
        LOG(info) << "Found generator " << gen << " in the list of available generators \n";
        if (gen.compare("boxgen") == 0) {
            if (mConfigs[index].compare("") == 0) {
              gens.push_back(std::make_unique<o2::eventgen::BoxGenerator>());
            } else {
              std::stringstream ss(mConfigs[index]);
              std::string pars;
              std::vector<double> params;
              while (std::getline(ss, pars, ',')) {
                params.push_back(std::stod(pars));
              }
              gens.push_back(std::make_unique<o2::eventgen::BoxGenerator>(int(params[0]), int(params[1]), params[2], params[3], params[4], params[5], params[6], params[7]));
            }
            mGens.push_back(gen);
        } else if (gen.compare(0, 7, "pythia8") == 0) {
            gens.push_back(std::make_unique<o2::eventgen::GeneratorPythia8>());
            mConfsPythia8.push_back(mConfigs[index]);
            mGens.push_back(gen);
        } else if (gen.compare("extkinO2") == 0){
            gens.push_back(std::make_unique<o2::eventgen::GeneratorFromO2Kine>(mConfigs[index].c_str()));
            mGens.push_back(gen);
        } else if (gen.compare("external") == 0) {
            if (mConfigs[index].compare("") == 0) {
              LOG(fatal) << "No configuration provided";
              exit(1);
            } else {
              std::stringstream ss(mConfigs[index]);
              std::string pars;
              std::vector<std::string> params;
              while (std::getline(ss, pars, ',')) {
                params.push_back(pars);
              }
              auto& extgen_filename = params[0];
              auto& extgen_func = params[1];
              auto extGen = std::unique_ptr<o2::eventgen::Generator>(o2::conf::GetFromMacro<o2::eventgen::Generator*>(extgen_filename, extgen_func, "FairGenerator*", "extgen"));
              if (!extGen) {
                LOG(fatal) << "Failed to load external generator from " << extgen_filename << " with function " << extgen_func;
                exit(1);
              }
              gens.push_back(std::move(extGen));
              mGens.push_back(gen);
            }
        } else if (gen.compare("hepmc") == 0) {
            if (mConfigs[index].compare("") == 0) {
              LOG(fatal) << "No configuration provided";
              exit(1);
            } else {
                std::stringstream ss(mConfigs[index]);
                std::string pars;
                std::vector<std::string> params;
                while (std::getline(ss, pars, ',')) {
                  params.push_back(pars);
                }
                gens.push_back(std::make_unique<o2::eventgen::GeneratorHepMC>());
                dynamic_cast<o2::eventgen::GeneratorHepMC*>(gens.back().get())->setFileNames(std::string(params[0]));
                dynamic_cast<o2::eventgen::GeneratorHepMC*>(gens.back().get())->setVersion(std::stoi(params[1]));
                mGens.push_back(gen);
            }
        } else {
              LOG(info) << "Generator " << gen << " not found in the list of available generators \n";
            }
      }
      index++;
    }  
  }

  Bool_t GeneratorHybrid::Init()
  {
    // init all sub-gens
    int count = 0;
    for (auto& gen : mGens)
    {
      if (gen == "pythia8"){
        auto config = std::string(std::getenv("O2_ROOT")) + mConfsPythia8[count];
        LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
        dynamic_cast<o2::eventgen::GeneratorPythia8*>(gens[count].get())->setConfig(config);
      } else if (gen == "pythia8pp") {
        auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_inel.cfg";
        LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
        dynamic_cast<o2::eventgen::GeneratorPythia8*>(gens[count].get())->setConfig(config);
      } else if (gen == "pythia8hf") {
        auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_hf.cfg";
        LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
        dynamic_cast<o2::eventgen::GeneratorPythia8*>(gens[count].get())->setConfig(config);
      } else if (gen == "pythia8hi") {
        auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_hi.cfg";
        LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
        dynamic_cast<o2::eventgen::GeneratorPythia8*>(gens[count].get())->setConfig(config);
      } else if (gen == "pythia8powheg") {
        auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_powheg.cfg";
        LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
        dynamic_cast<o2::eventgen::GeneratorPythia8*>(gens[count].get())->setConfig(config);
      }
      gens[count]->Init();
      addSubGenerator(count, gen);
      count++;  
    }
    return Generator::Init();
  }

  Bool_t GeneratorHybrid::generateEvent()
  {
      // Order randomisation or sequence of generators
      // following provided fractions, if not generators are used in proper sequence
      if (mRandomize) {
        mIndex = gRandom->Integer(mGens.size());
      } else {
        while (mFractions[mCurrentFraction] == 0 || mseqCounter == mFractions[mCurrentFraction]) {
          if (mFractions[mCurrentFraction] != 0)
            mseqCounter = 0;
          mCurrentFraction = (mCurrentFraction + 1) % mFractions.size();
        }
        mIndex = mCurrentFraction;
      }
      LOG(info) << "GeneratorHybrid: generating event with generator " << mGens[mIndex];
      gens[mIndex]->clearParticles(); // clear container of this class
      gens[mIndex]->generateEvent();
      // notify the sub event generator
      notifySubGenerator(mIndex);
      mseqCounter++;
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