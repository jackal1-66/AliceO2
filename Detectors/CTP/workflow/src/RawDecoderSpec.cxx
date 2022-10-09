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

#include <string>
#include <fairlogger/Logger.h>
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"
#include "CTPWorkflow/RawDecoderSpec.h"
#include "CommonUtils/VerbosityConfig.h"

using namespace o2::ctp::reco_workflow;

void RawDecoderSpec::init(framework::InitContext& ctx)
{
  mNTFToIntegrate = ctx.options().get<int>("ntf-to-average");
  mVerbose = ctx.options().get<bool>("use-verbose-mode");
}

void RawDecoderSpec::run(framework::ProcessingContext& ctx)
{
  mOutputDigits.clear();
  std::map<o2::InteractionRecord, CTPDigit> digits;
  const gbtword80_t bcidmask = 0xfff;
  gbtword80_t pldmask;
  using InputSpec = o2::framework::InputSpec;
  using ConcreteDataTypeMatcher = o2::framework::ConcreteDataTypeMatcher;
  using Lifetime = o2::framework::Lifetime;

  // setUpDummyLink
  auto& inputs = ctx.inputs();
  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
    std::vector<InputSpec> dummy{InputSpec{"dummy", o2::framework::ConcreteDataMatcher{"CTP", "RAWDATA", 0xDEADBEEF}}};
    for (const auto& ref : o2::framework::InputRecordWalker(inputs, dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(ref);
      if (payloadSize == 0) {
        auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
        if (++contDeadBeef <= maxWarn) {
          LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
               dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
               contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
        }
        if (mDoDigits) {
          ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
        }
        if (mDoLumi) {
          ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe}, mOutputLumiInfo);
        }
        return;
      }
    }
    contDeadBeef = 0; // if good data, reset the counter
  }
  //
  std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, Lifetime::Timeframe}};
  o2::framework::DPLRawParser parser(ctx.inputs(), filter);
  std::vector<LumiInfo> lumiPointsHBF1;
  uint64_t countsMB = 0;
  uint32_t payloadCTP;
  uint32_t orbit0 = 0;
  bool first = true;
  gbtword80_t remnant = 0;
  uint32_t size_gbt = 0;
  for (auto it = parser.begin(); it != parser.end(); ++it) {
    auto rdh = it.get_if<o2::header::RAWDataHeader>();
    auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
    // std::cout << "==================>" << std::hex << triggerOrbit << std::endl;
    if (first) {
      orbit0 = triggerOrbit;
      first = false;
    }
    auto feeID = o2::raw::RDHUtils::getFEEID(rdh); // 0 = IR, 1 = TCR
    auto linkCRU = (feeID & 0xf00) >> 8;
    if (linkCRU == o2::ctp::GBTLinkIDIntRec) {
      payloadCTP = o2::ctp::NIntRecPayload;
    } else if (linkCRU == o2::ctp::GBTLinkIDClassRec) {
      payloadCTP = o2::ctp::NClassPayload;
      if (!mDoDigits) {
        continue;
      }
    } else {
      LOG(error) << "Unxpected  CTP CRU link:" << linkCRU;
    }
    LOG(debug) << "RDH FEEid: " << feeID << " CTP CRU link:" << linkCRU << " Orbit:" << triggerOrbit;
    pldmask = 0;
    for (uint32_t i = 0; i < payloadCTP; i++) {
      pldmask[12 + i] = 1;
    }
    // LOG(info) << "pldmask:" << pldmask;
    //  TF in 128 bits words
    gsl::span<const uint8_t> payload(it.data(), it.size());
    gbtword80_t gbtWord = 0;
    int wordCount = 0;
    std::vector<gbtword80_t> diglets;
    if (orbit0 != triggerOrbit) {
      if (mDoLumi && payloadCTP == o2::ctp::NIntRecPayload) { // create lumi per HB
        lumiPointsHBF1.emplace_back(LumiInfo{triggerOrbit, 0, countsMB});
        countsMB = 0;
      }
      remnant = 0;
      size_gbt = 0;
      orbit0 = triggerOrbit;
    }
    for (auto payloadWord : payload) {
      // LOG(info) << wordCount << " payload:" << int(payloadWord);
      if (wordCount == 15) {
        wordCount = 0;
      } else if (wordCount > 9) {
        wordCount++;
      } else if (wordCount == 9) {
        for (int i = 0; i < 8; i++) {
          gbtWord[wordCount * 8 + i] = bool(int(payloadWord) & (1 << i));
        }
        wordCount++;
        diglets.clear();
        // LOG(info) << " gbtword:" << gbtWord;
        makeGBTWordInverse(diglets, gbtWord, remnant, size_gbt, payloadCTP);
        // save digit in buffer recs
        for (auto diglet : diglets) {
          if (mDoLumi && payloadCTP == o2::ctp::NIntRecPayload) {
            gbtword80_t pld = (diglet & mTVXMask);
            if (pld.count() != 0) {
              countsMB++;
            }
          }
          if (!mDoDigits) {
            continue;
          }
          gbtword80_t pld = (diglet & pldmask);
          if (pld.count() == 0) {
            continue;
          }
          // LOG(info) << "    pld:" << pld;
          pld >>= 12;
          CTPDigit digit;
          uint32_t bcid = (diglet & bcidmask).to_ulong();
          o2::InteractionRecord ir;
          ir.orbit = triggerOrbit;
          ir.bc = bcid;
          digit.intRecord = ir;
          if (linkCRU == o2::ctp::GBTLinkIDIntRec) {
            LOG(debug) << "InputMaskCount:" << digits[ir].CTPInputMask.count();
            if (digits.count(ir) == 0) {
              digit.setInputMask(pld);
              digits[ir] = digit;
              LOG(debug) << bcid << " inputs case 0 bcid orbit " << triggerOrbit << " pld:" << pld;
            } else if (digits.count(ir) == 1) {
              if (digits[ir].CTPInputMask.count() == 0) {
                digits[ir].setInputMask(pld);
                LOG(debug) << bcid << " inputs bcid vase 1 orbit " << triggerOrbit << " pld:" << pld;
              } else {
                LOG(error) << "Two CTP IRs with the same timestamp.";
              }
            } else {
              LOG(error) << "Two digits with the same rimestamp.";
            }
          } else if (linkCRU == o2::ctp::GBTLinkIDClassRec) {
            if (digits.count(ir) == 0) {
              digit.setClassMask(pld);
              digits[ir] = digit;
              LOG(debug) << bcid << " class bcid case 0 orbit " << triggerOrbit << " pld:" << pld;
            } else if (digits.count(ir) == 1) {
              if (digits[ir].CTPClassMask.count() == 0) {
                digits[ir].setClassMask(pld);
                LOG(debug) << bcid << " class bcid case 1 orbit " << triggerOrbit << " pld:" << pld;
              } else {
                LOG(error) << "Two CTP Class masks for same timestamp";
              }
            } else {
            }
          } else {
            LOG(error) << "Unxpected  CTP CRU link:" << linkCRU;
          }
        }
        gbtWord = 0;
      } else {
        for (int i = 0; i < 8; i++) {
          gbtWord[wordCount * 8 + i] = bool(int(payloadWord) & (1 << i));
        }
        wordCount++;
      }
    }
  }
  if (mDoDigits) {
    for (auto const digmap : digits) {
      mOutputDigits.push_back(digmap.second);
    }
    LOG(info) << "[CTPRawToDigitConverter - run] Writing " << mOutputDigits.size() << " digits ...";
    ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
  }
  if (mDoLumi) {
    lumiPointsHBF1.emplace_back(LumiInfo{orbit0, 0, countsMB});
    uint32_t tfCounts = 0;
    for (auto const& lp : lumiPointsHBF1) {
      tfCounts += lp.counts;
    }
    mHistory.push_back(tfCounts);
    mCounts += tfCounts;
    if (mHistory.size() <= mNTFToIntegrate) {
      mNHBIntegrated += lumiPointsHBF1.size();
    } else {
      mCounts -= mHistory.front();
      mHistory.pop_front();
    }
    if (mNHBIntegrated) {
      mOutputLumiInfo.orbit = lumiPointsHBF1[0].orbit;
    }
    mOutputLumiInfo.counts = mCounts;
    mOutputLumiInfo.nHBFCounted = mNHBIntegrated;
    if (mVerbose) {
      LOGP(info, "Orbit {}: {} counts in {} HBFs -> lumi = {:.3e}+-{:.3e}", mOutputLumiInfo.orbit, mCounts, mNHBIntegrated, mOutputLumiInfo.getLumi(), mOutputLumiInfo.getLumiError());
    }
    ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe}, mOutputLumiInfo);
  }
}

// Inverse of Digits2Raw::makeGBTWord
void RawDecoderSpec::makeGBTWordInverse(std::vector<gbtword80_t>& diglets, gbtword80_t& GBTWord, gbtword80_t& remnant, uint32_t& size_gbt, uint32_t Npld)
{
  gbtword80_t diglet = remnant;
  uint32_t i = 0;
  while (i < (NGBT - Npld)) {
    std::bitset<NGBT> masksize = 0;
    for (uint32_t j = 0; j < (Npld - size_gbt); j++) {
      masksize[j] = 1;
    }
    diglet |= (GBTWord & masksize) << (size_gbt);
    diglets.push_back(diglet);
    diglet = 0;
    i += Npld - size_gbt;
    GBTWord = GBTWord >> (Npld - size_gbt);
    size_gbt = 0;
  }
  size_gbt = NGBT - i;
  remnant = GBTWord;
}
o2::framework::DataProcessorSpec o2::ctp::reco_workflow::getRawDecoderSpec(bool askDISTSTF, bool digits, bool lumi)
{
  if (!digits && !lumi) {
    throw std::runtime_error("all outputs were disabled");
  }
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, o2::framework::Lifetime::Optional);
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }

  std::vector<o2::framework::OutputSpec> outputs;
  if (digits) {
    outputs.emplace_back("CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  }
  if (lumi) {
    outputs.emplace_back("CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe);
  }
  return o2::framework::DataProcessorSpec{
    "ctp-raw-decoder",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::ctp::reco_workflow::RawDecoderSpec>(digits, lumi)},
    o2::framework::Options{
      {"ntf-to-average", o2::framework::VariantType::Int, 90, {"Time interval for averaging luminosity in units of TF"}},
      {"use-verbose-mode", o2::framework::VariantType::Bool, false, {"Verbose logging"}}}};
}