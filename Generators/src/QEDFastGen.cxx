// Copyright 2024-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author M+Giacalone - April 2026
/// \brief Standalone generator for QED processes using a fast simulation approach based on ONNX models.

#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

// ROOT
#include <TFile.h>
#include <TGrid.h>
#include <TTree.h>
#include <TVector3.h>
#include "TSystem.h"

// CCDB
#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"

// AliceO2 hit types
#include "DataFormatsFDD/Hit.h"
#include "DataFormatsFT0/HitType.h"
#include "DataFormatsFV0/Hit.h"
#include "ITSMFTSimulation/Hit.h"

// AliceO2 naming conventions
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "CommonUtils/NameConf.h"
#include "MathUtils/Cartesian.h"  // o2::math_utils::Point3D / Vector3D

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <thread>
#include <utility>
#include <boost/program_options.hpp>

using json = nlohmann::json;

// -- QuantileTransformer (inverse only)
struct QuantileScaler {
  std::vector<float> references;                     // size Q, sorted ascending
  std::vector<std::vector<float>> quantiles_by_feat; // [n_feat][Q]
  int n_features{};

  static float interp(const std::vector<float>& xs,
                      const std::vector<float>& ys, float z)
  {
    const int n = static_cast<int>(xs.size());
    if (z <= xs[0])     return ys[0];
    if (z >= xs[n - 1]) return ys[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      (xs[mid] <= z ? lo : hi) = mid;
    }
    const float t = (z - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + t * (ys[hi] - ys[lo]);
  }

  void inverse(std::vector<float>& row) const
  {
    // Apply the inverse transformation using the quantiles and references.
    // std::erff is the error function and estimates the probability that a measurement with a normal distribution (subject to errors) is within
    // standard deviations of the mean
    constexpr float INV_SQRT2 = 1. / std::sqrt(2.f);
    for (int i = 0; i < n_features; ++i) {
      const float u = 0.5f * (1.f + std::erff(row[i] * INV_SQRT2));
      row[i] = interp(references, quantiles_by_feat[i], u);
    }
  }
};

// -- Hit-count PMF
struct HitPmf {
  std::vector<int> vals;
  std::vector<double> probs;
};

// -- ONNX Runtime session wrapper
struct OrtSession {
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names_str;
  std::vector<std::string> output_names_str;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
};

// -- All artefacts retrieved from disk
struct Artefacts {
  std::vector<std::string> detectors;
  int n_ctx{};
  int cfm_steps{};  // Euler integration steps read from model_cfg.json
  std::map<std::string, int> n_features;
  std::map<std::string, OrtSession> sessions;
  std::map<std::string, QuantileScaler> scalers;
  std::map<std::string, std::vector<std::string>> features;
  // CDF: maps each detector name to a map from pattern string (e.g. "DET1,DET2,...") to HitPmf
  std::map<std::string, std::map<std::string, HitPmf>> cdfs;
  std::vector<std::vector<std::string>> patterns;
  std::vector<double> probs;
};

// -- Helpers

// Simply makes a comma-separated string from a list of detectors, sorted alphabetically.
static std::string make_key(std::vector<std::string> active)
{
  std::sort(active.begin(), active.end());
  std::string key;
  for (size_t i = 0; i < active.size(); ++i) {
    if (i)
      key += ',';
    key += active[i];
  }
  return key;
}

// Checks if the pattern contains a given detector
static bool key_has(const std::string& key, const std::string& det)
{
  size_t pos = 0;
  while (pos < key.size()) {
    size_t comma = key.find(',', pos);
    std::string tok = (comma == std::string::npos) ? key.substr(pos) : key.substr(pos, comma - pos);
    if (tok == det)
      return true;
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return false;
}

static Ort::Env& ort_env()
{
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "qedFastSim");
  return env;
}

// -- Download ONNX models listed in models_onnx.json
// The JSON maps detector names to their source paths, e.g.:
//   { "FDD": "alien://path/to/cfm_FDD.onnx", "FT0": "ccdb://Users/.../cfm_FT0", ... }
// Each model is downloaded into <models_dir>/cfm_<DET>.onnx, which is the path
// that load_artefacts() expects.  Local paths are ignored (nothing to download).
static void download_models(const std::string& dir)
{
  const std::string json_path = dir + "models_onnx.json";
  std::ifstream f(json_path);
  if (!f) {
    // No models_onnx.json present: assume all .onnx files are already local.
    return;
  }
  const json j = json::parse(f);

  struct Entry { std::string det; std::string src; std::string local; };
  std::vector<Entry> alien_entries, ccdb_entries;

  for (const auto& [det, val] : j.items()) {
    const std::string src = val.get<std::string>();
    // ONNX files are saved in current folder
    const std::string local = "cfm_" + det + ".onnx";
    if (src.starts_with("alien://"))
      alien_entries.push_back({det, src, local});
    else if (src.starts_with("ccdb://"))
      ccdb_entries.push_back({det, src, local});
    // else: already a local path, nothing to download
  }

  if (alien_entries.empty() && ccdb_entries.empty())
    return;

  LOG(info) << "Downloading ONNX models ...\n";

  // Alien downloads via TGrid
  if (!alien_entries.empty()) {
    if (!gGrid) {
      TGrid::Connect("alien://");
      if (!gGrid) {
        LOG(fatal) << "AliEn connection failed, check token.";
        exit(1);
      }  
    }
    for (const auto& e : alien_entries) {
      LOG(info) << "  [alien] " << e.src << " -> " << e.local << "\n";
      if (!TFile::Cp(e.src.c_str(), e.local.c_str())) {
        LOG(fatal) << "Error: Model file " << e.src << " does not exist or failed to copy!";
        exit(1);
      }  
    }
  }

  // CCDB downloads via CcdbApi
  if (!ccdb_entries.empty()) {
    o2::ccdb::CcdbApi api;
    api.init("http://alice-ccdb.cern.ch");
    const long ts = o2::ccdb::getCurrentTimestamp();
    std::map<std::string, std::string> filter;
    for (const auto& e : ccdb_entries) {
      auto ccdb_path = e.src.substr(7); // strip "ccdb://"
      // CCDB uses folder paths; strip .onnx filename if it was included
      if (ccdb_path.find(".onnx") != std::string::npos)
        ccdb_path = ccdb_path.substr(0, ccdb_path.find_last_of('/'));
      LOG(info) << "  [ccdb]  " << ccdb_path << " -> " << e.local << "\n";
      const std::string local_fname = "cfm_" + e.det + ".onnx";
      if (!api.retrieveBlob(ccdb_path, "./", filter, ts, false, local_fname.c_str())) {
        LOG(fatal) << "Error: Failed to retrieve " << ccdb_path << " from CCDB!";
        exit(1);
      }  
    }
  }
}

// -- Load artefacts
static Artefacts load_artefacts(const std::string& dir, int n_threads)
{
  Artefacts art;

  auto open_json = [&](const std::string& name) -> json {
    std::ifstream f(dir + name);
    if (!f) {
      LOG(fatal) << "Error: Cannot open " << dir << name;
      exit(1);
    }  
    return json::parse(f);
  };
  // Using limited scopes for each file to ensure ifstream objects are closed after reading
  // model_cfg.json
  {
    auto j = open_json("model_cfg.json");
    art.detectors = j["fileNames"].get<std::vector<std::string>>();
    art.n_ctx = j["N_CTX"].get<int>();
    for (const auto& det : art.detectors)
      art.n_features[det] = j["n_features"][det].get<int>();
    art.cfm_steps = j.value("CFM_STEPS", 10);
  }

  // scalers.json
  {
    auto j = open_json("scalers.json");
    for (auto& [det, v] : j.items()) {
      QuantileScaler sc;
      sc.references = v["references"].get<std::vector<float>>();
      // JSON layout: quantiles[q_idx][feat_idx] transposed to quantiles_by_feat[feat_idx][q_idx]
      auto q_raw = v["quantiles"].get<std::vector<std::vector<float>>>();
      const int n_q = static_cast<int>(q_raw.size());
      const int n_f = n_q > 0 ? static_cast<int>(q_raw[0].size()) : 0;
      sc.n_features = n_f;
      sc.quantiles_by_feat.resize(n_f);
      for (int f = 0; f < n_f; ++f) {
        sc.quantiles_by_feat[f].resize(n_q);
        for (int q = 0; q < n_q; ++q)
          sc.quantiles_by_feat[f][q] = q_raw[q][f];
      }
      art.scalers[det] = std::move(sc);
    }
  }

  // flow_features.json
  {
    auto j = open_json("flow_features.json");
    for (auto& [det, v] : j.items())
      art.features[det] = v.get<std::vector<std::string>>();
  }

  // hit_count_cdfs.json
  {
    auto j = open_json("hit_count_cdfs.json");
    for (auto& [det, pat_map] : j.items())
      for (auto& [pat, entry] : pat_map.items()) {
        HitPmf pmf;
        pmf.vals = entry["vals"].get<std::vector<int>>();
        pmf.probs = entry["probs"].get<std::vector<double>>();
        art.cdfs[det][pat] = std::move(pmf);
      }
  }

  // patterns.json
  {
    auto j = open_json("patterns.json");
    art.patterns = j["patterns"].get<std::vector<std::vector<std::string>>>();
    art.probs = j["probs"].get<std::vector<double>>();
  }

  // ONNX sessions
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(n_threads);
  opts.SetInterOpNumThreads(n_threads);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  LOG(info) << "  Loading ONNX sessions:\n";
  // ONNX files are expected to be in the current folder with names cfm_<DET>.onnx, as ensured by download_models()
  for (const auto& det : art.detectors) {
    std::string path = "cfm_" + det + ".onnx";
    OrtSession os;
    os.session = std::make_unique<Ort::Session>(ort_env(), path.c_str(), opts);
    for (size_t i = 0; i < os.session->GetInputCount(); ++i) {
      auto name = os.session->GetInputNameAllocated(i, os.allocator);
      os.input_names_str.emplace_back(name.get());
    }
    for (size_t i = 0; i < os.session->GetOutputCount(); ++i) {
      auto name = os.session->GetOutputNameAllocated(i, os.allocator);
      os.output_names_str.emplace_back(name.get());
    }
    for (const auto& s : os.input_names_str)
      os.input_names.push_back(s.c_str());
    for (const auto& s : os.output_names_str)
      os.output_names.push_back(s.c_str());

    LOG(info) << "    ok  " << det << "\n";
    art.sessions[det] = std::move(os);
  }

  return art;
}

// -- Simulation
// Event: maps each detector name to a list of hit feature-vectors [n_hits][n_features]
using Event = std::map<std::string, std::vector<std::vector<float>>>;

static std::vector<Event> simulate(int n_events, const Artefacts& art, std::mt19937& rng)
{
  const auto& dets = art.detectors;

  // Stage 1: sample coincidence pattern for every event
  // std::discrete_distribution is a random number distribution that produces integers on the interval [0, n) according to the discrete probability distribution defined by the provided weights (probs).
  // Unfortunately I didn't find a way in ROOT to do this, so using the STL version.
  // This samples which detectors are active
  std::discrete_distribution<int> pat_dist(art.probs.begin(), art.probs.end());
  std::vector<int> chosen(n_events);
  for (int i = 0; i < n_events; ++i)
    chosen[i] = pat_dist(rng);

  // Build context row cache (one row per unique pattern)
  std::map<std::string, std::vector<float>> ctx_cache;
  for (const auto& pat : art.patterns) {
    std::string key = make_key(pat);
    if (ctx_cache.count(key))
      continue;
    std::vector<float> flags;
    flags.reserve(dets.size());
    for (const auto& d : dets)
      flags.push_back(std::find(pat.begin(), pat.end(), d) != pat.end() ? 1.f : 0.f);
    ctx_cache[key] = std::move(flags);
  }

  // Stage 2: sample hit counts per active detector.
  struct HitEntry { int ev_idx; int n_hits; std::string pkey; };
  std::map<std::string, std::vector<HitEntry>> det_entries;

  for (int i = 0; i < n_events; ++i) {
    const auto& pat = art.patterns[chosen[i]];
    if (pat.empty())
      continue; // empty pattern – no hits for any detector
    std::string pkey = make_key(pat);

    for (const auto& det : dets) {
      bool active = std::find(pat.begin(), pat.end(), det) != pat.end();
      if (!active || !art.sessions.count(det))
        continue;

      const auto& cdf_map = art.cdfs.at(det);
      const HitPmf* pmf_ptr = nullptr;
      {
        auto it = cdf_map.find(pkey);
        if (it != cdf_map.end() && !it->second.vals.empty())
          pmf_ptr = &it->second;
        else {
          for (const auto& [k, v] : cdf_map)
            if (!v.vals.empty() && key_has(k, det)) {
              pmf_ptr = &v;
              break;
            }
        }
      }
      if (!pmf_ptr)
        continue;

      // Sample the number of hits for this detector in this event using the probability mass function (PMF)
      std::discrete_distribution<int> pmf_dist(pmf_ptr->probs.begin(), pmf_ptr->probs.end());
      int n_hits = pmf_ptr->vals[pmf_dist(rng)];
      if (n_hits == 0)
        continue;

      det_entries[det].push_back({i, n_hits, pkey});
    }
  }

  // Stage 3: Euler ODE -- x += velocity(x, t, ctx)*dt for cfm_steps steps.
  const float dt = 1.0f / static_cast<float>(art.cfm_steps);

  // Normal distribution provides the starting noise for the features
  std::normal_distribution<float> normal(0.f, 1.f);
  std::map<std::string, std::map<int, std::vector<std::vector<float>>>> all_hits;
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  for (const auto& det : dets) {
    if (!det_entries.count(det) || det_entries.at(det).empty())
      continue;

    const auto& entries          = det_entries.at(det);
    const QuantileScaler& sc     = art.scalers.at(det);
    int n_feat                   = art.n_features.at(det);
    const OrtSession& os         = art.sessions.at(det);

    int total = 0;
    for (const auto& e : entries)
      total += e.n_hits;

    std::vector<float> x_data(total * n_feat);
    for (auto& v : x_data)
      v = normal(rng);

    std::vector<float> ctx_data(total * art.n_ctx);
    {
      int off = 0;
      for (const auto& e : entries) {
        const std::vector<float>& ctx_row = ctx_cache.at(e.pkey);
        for (int r = 0; r < e.n_hits; ++r)
          std::copy(ctx_row.begin(), ctx_row.end(), ctx_data.begin() + (off + r) * art.n_ctx);
        off += e.n_hits;
      }
    }

    std::vector<float> t_vec(total);

    std::array<int64_t, 2> x_shape{total, n_feat};
    std::array<int64_t, 1> t_shape{total};
    std::array<int64_t, 2> c_shape{total, art.n_ctx};

    for (int step = 0; step < art.cfm_steps; ++step) {
      std::fill(t_vec.begin(), t_vec.end(), step * dt);

      Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
        mem_info, x_data.data(), static_cast<size_t>(total * n_feat),
        x_shape.data(), x_shape.size());
      Ort::Value t_tensor = Ort::Value::CreateTensor<float>(
        mem_info, t_vec.data(), static_cast<size_t>(total),
        t_shape.data(), t_shape.size());
      Ort::Value c_tensor = Ort::Value::CreateTensor<float>(
        mem_info, ctx_data.data(), static_cast<size_t>(total * art.n_ctx),
        c_shape.data(), c_shape.size());

      std::vector<Ort::Value> inputs;
      inputs.push_back(std::move(x_tensor));
      inputs.push_back(std::move(t_tensor));
      inputs.push_back(std::move(c_tensor));

      auto out = os.session->Run(
        Ort::RunOptions{nullptr},
        os.input_names.data(), inputs.data(), inputs.size(),
        os.output_names.data(), os.output_names.size());

      const float* vel = out[0].GetTensorData<float>();
      const int sz = total * n_feat;
      for (int k = 0; k < sz; ++k)
        x_data[k] += vel[k] * dt;
    }

    const float* hits_ptr = x_data.data();
    int offset = 0;
    for (const auto& e : entries) {
      std::vector<std::vector<float>> ev_hits;
      ev_hits.reserve(e.n_hits);
      for (int r = 0; r < e.n_hits; ++r) {
        std::vector<float> hit(hits_ptr + (offset + r) * n_feat,
                               hits_ptr + (offset + r) * n_feat + n_feat);
        sc.inverse(hit);
        ev_hits.push_back(std::move(hit));
      }
      all_hits[det][e.ev_idx] = std::move(ev_hits);
      offset += e.n_hits;
    }
  }

  // Assemble per-event result maps
  std::vector<Event> results(n_events);
  for (int i = 0; i < n_events; ++i)
    for (const auto& det : dets)
      if (all_hits.count(det) && all_hits.at(det).count(i))
        results[i][det] = std::move(all_hits[det][i]);

  return results;
}

// -- Convert CFM features to FDD Hit
static o2::fdd::Hit make_fdd_hit(int trackID, const std::vector<float>& f)
{
  TVector3 pos(f[0], f[1], f[2]);
  int nPhot = static_cast<int>(std::lround(f[5]));
  if (nPhot < 0)
    nPhot = 0;
  return o2::fdd::Hit(trackID, 0, pos, static_cast<double>(f[3]),
                      static_cast<double>(f[4]), nPhot);
}

// --Convert CFM features to FT0 HitType
static o2::ft0::HitType make_ft0_hit(int trackID, const std::vector<float>& f)
{
  return o2::ft0::HitType(f[0], f[1], f[2],
                          f[3], f[4],
                          trackID, 0);
}

// -- Convert CFM features to FV0 Hit
// NOTE: FV0Hit.mTime is stored in seconds in O2 ROOT files.  The CFM scaler was
// trained with mTime converted to nanoseconds (to avoid the sklearn
// QuantileTransformer BOUNDS_THRESHOLD=1e-7 clipping bug).  Convert back here.
static o2::fv0::Hit make_fv0_hit(int trackID, const std::vector<float>& f)
{
  o2::math_utils::Point3D<float> startPos(f[8], f[9], f[10]);
  o2::math_utils::Point3D<float> endPos(f[0], f[1], f[2]);
  o2::math_utils::Vector3D<float> startMom(f[5], f[6], f[7]);
  return o2::fv0::Hit(trackID, 0,
                      startPos, endPos, startMom,
                      static_cast<double>(f[11]),         // startE
                      static_cast<double>(f[3]) * 1e-9,  // endTime  (ns to s)
                      static_cast<double>(f[4]),          // eLoss
                      11);                                // default PDG: electron
}

// -- Convert CFM features to ITSMFT Hit (shared by ITS and MFT)
// NOTE: ITSHit/MFTHit.mTime is stored in seconds in O2 ROOT files.  Same
// ns-to-s back-conversion as FV0 (see above).
static o2::itsmft::Hit make_itsmft_hit(int trackID, const std::vector<float>& f)
{
  TVector3 startPos(f[8], f[9], f[10]);
  TVector3 endPos(f[0], f[1], f[2]);
  TVector3 mom(f[5], f[6], f[7]);
  return o2::itsmft::Hit(trackID, 0,
                         startPos, endPos, mom,
                         static_cast<double>(f[11]),        // startE
                         static_cast<double>(f[3]) * 1e-9, // endTime  (ns to s)
                         static_cast<double>(f[4]),         // eLoss
                         o2::itsmft::Hit::kTrackEntering,   // startStatus
                         o2::itsmft::Hit::kTrackExiting);   // endStatus
}

// -- Write per-detector Hits.root files
static void write_hits(
  const std::vector<Event>& events,
  const Artefacts& art,
  const std::string& out_dir,
  const std::string& prefix)
{
  std::vector<o2::fdd::Hit> fdd_hits;
  std::vector<o2::ft0::HitType> ft0_hits;
  std::vector<o2::fv0::Hit> fv0_hits;
  std::vector<o2::itsmft::Hit> its_hits;
  std::vector<o2::itsmft::Hit> mft_hits;

  auto open_file = [&](const std::string& fname) -> std::pair<TFile*, TTree*> {
    TFile* f = TFile::Open(fname.c_str(), "RECREATE");
    if (!f || f->IsZombie()) {
      LOG(fatal) << "Error: Cannot create file " << fname;
      exit(1);
    }  
    auto* t = new TTree("o2sim", "o2sim");
    return {f, t};
  };

  const bool has_fdd = std::count(art.detectors.begin(), art.detectors.end(), "FDD");
  const bool has_ft0 = std::count(art.detectors.begin(), art.detectors.end(), "FT0");
  const bool has_fv0 = std::count(art.detectors.begin(), art.detectors.end(), "FV0");
  const bool has_its = std::count(art.detectors.begin(), art.detectors.end(), "ITS");
  const bool has_mft = std::count(art.detectors.begin(), art.detectors.end(), "MFT");

  TFile* fdd_file = nullptr;
  TTree* fdd_tree = nullptr;
  if (has_fdd) {
    std::string fname = out_dir + prefix + "_HitsFDD.root";
    std::tie(fdd_file, fdd_tree) = open_file(fname);
    fdd_tree->Branch("FDDHit", &fdd_hits);
  }

  TFile* ft0_file = nullptr;
  TTree* ft0_tree = nullptr;
  if (has_ft0) {
    std::string fname = out_dir + prefix + "_HitsFT0.root";
    std::tie(ft0_file, ft0_tree) = open_file(fname);
    ft0_tree->Branch("FT0Hit", &ft0_hits);
  }

  TFile* fv0_file = nullptr;
  TTree* fv0_tree = nullptr;
  if (has_fv0) {
    std::string fname = out_dir + prefix + "_HitsFV0.root";
    std::tie(fv0_file, fv0_tree) = open_file(fname);
    fv0_tree->Branch("FV0Hit", &fv0_hits);
  }

  TFile* its_file = nullptr;
  TTree* its_tree = nullptr;
  if (has_its) {
    std::string fname = out_dir + prefix + "_HitsITS.root";
    std::tie(its_file, its_tree) = open_file(fname);
    its_tree->Branch("ITSHit", &its_hits);
  }

  TFile* mft_file = nullptr;
  TTree* mft_tree = nullptr;
  if (has_mft) {
    std::string fname = out_dir + prefix + "_HitsMFT.root";
    std::tie(mft_file, mft_tree) = open_file(fname);
    mft_tree->Branch("MFTHit", &mft_hits);
  }

  for (int ev = 0; ev < static_cast<int>(events.size()); ++ev) {
    const Event& event = events[ev];
    const int trackID = 0;

    // FDD
    if (has_fdd) {
      fdd_hits.clear();
      auto it = event.find("FDD");
      if (it != event.end()) {
        for (const auto& feat : it->second)
          fdd_hits.push_back(make_fdd_hit(trackID, feat));
      }
      fdd_file->cd();
      fdd_tree->Fill();
    }

    // FT0
    if (has_ft0) {
      ft0_hits.clear();
      auto it = event.find("FT0");
      if (it != event.end()) {
        for (const auto& feat : it->second)
          ft0_hits.push_back(make_ft0_hit(trackID, feat));
      }
      ft0_file->cd();
      ft0_tree->Fill();
    }

    // FV0
    if (has_fv0) {
      fv0_hits.clear();
      auto it = event.find("FV0");
      if (it != event.end()) {
        for (const auto& feat : it->second)
          fv0_hits.push_back(make_fv0_hit(trackID, feat));
      }
      fv0_file->cd();
      fv0_tree->Fill();
    }

    // ITS
    if (has_its) {
      its_hits.clear();
      auto it = event.find("ITS");
      if (it != event.end()) {
        for (const auto& feat : it->second)
          its_hits.push_back(make_itsmft_hit(trackID, feat));
      }
      its_file->cd();
      its_tree->Fill();
    }

    // MFT
    if (has_mft) {
      mft_hits.clear();
      auto it = event.find("MFT");
      if (it != event.end()) {
        for (const auto& feat : it->second)
          mft_hits.push_back(make_itsmft_hit(trackID, feat));
      }
      mft_file->cd();
      mft_tree->Fill();
    }
  }

  auto close_file = [](TFile* f, TTree* t) {
    if (!f)
      return;
    f->cd();
    t->Write("", TObject::kWriteDelete);
    f->Close();
    delete f;
  };

  close_file(fdd_file, fdd_tree);
  close_file(ft0_file, ft0_tree);
  close_file(fv0_file, fv0_tree);
  close_file(its_file, its_tree);
  close_file(mft_file, mft_tree);
}

// -- Print summary
static void print_summary(const std::vector<Event>& events, const Artefacts& art)
{
  LOG(info) << "\n  Per-detector statistics:\n";
  for (const auto& det : art.detectors) {
    int n_active = 0, n_hits = 0;
    for (const auto& ev : events) {
      auto it = ev.find(det);
      if (it != ev.end() && !it->second.empty()) {
        ++n_active;
        n_hits += static_cast<int>(it->second.size());
      }
    }
    LOG(info) << "    " << det
              << "  active_events=" << n_active
              << "/" << events.size()
              << "  total_hits=" << n_hits << "\n";
  }
}

// -- Entry point ---------------------------------------------------------------
namespace bpo = boost::program_options;

int main(int argc, char* argv[])
{
  const int hw = static_cast<int>(std::thread::hardware_concurrency());
  const int default_threads = std::max(1, std::min(8, hw));

  bpo::options_description options("qedFastSimCFM options");
  options.add_options()
    ("models-dir,m", bpo::value<std::string>()->default_value("${O2_ROOT}/share/Generators/QEDFastGen"),
                                                                        "directory with CFM artefacts from fastsim_cfm.ipynb")
    ("nevents,n",    bpo::value<int>()->required(),                     "number of events to simulate")
    ("output-dir,o", bpo::value<std::string>()->default_value("./"),    "directory where Hits.root files are written")
    ("prefix,p",     bpo::value<std::string>()->default_value("qed"), "file-name prefix")
    ("seed,s",       bpo::value<uint32_t>()->default_value(42u),        "RNG seed")
    ("threads,j",    bpo::value<int>()->default_value(default_threads), "ORT intra/inter-op thread count")
    ("help,h",       "produce help message");

  bpo::variables_map vm;
  try {
    bpo::store(bpo::parse_command_line(argc, argv, options), vm);
    if (vm.count("help")) {
      LOG(info) << options << "\n"
                << "Output files:\n"
                << "  <prefix>_HitsFDD.root  <prefix>_HitsFT0.root  <prefix>_HitsFV0.root\n"
                << "  <prefix>_HitsITS.root  <prefix>_HitsMFT.root\n";
      return 0;
    }
    bpo::notify(vm);
  } catch (const bpo::error& e) {
    LOG(fatal) << "Error parsing command-line arguments: " << e.what() << "\n\n" << options;
    return 1;
  }

  auto ensure_slash = [](std::string s) {
    if (!s.empty() && s.back() != '/')
      s += '/';
    return s;
  };

  // Tried to do this directly, but ROOT's TGrid::ExpandPathName returns a char* that must be freed with delete[], 
  // otherwise we get memory leaks
  auto expand = [](const std::string& s) -> std::string {
    char* p = gSystem->ExpandPathName(s.c_str());
    std::string result(p);
    delete[] p;
    return result;
  };

  const std::string models_dir = expand(ensure_slash(vm["models-dir"].as<std::string>()));
  const int n_events            = vm["nevents"].as<int>();
  const std::string out_dir    = expand(ensure_slash(vm["output-dir"].as<std::string>()));
  const std::string prefix      = vm["prefix"].as<std::string>();
  const uint32_t seed           = vm["seed"].as<uint32_t>();
  const int n_threads           = std::max(1, vm["threads"].as<int>());

  LOG(info) << "qedFastSimCFM - QED Conditional Flow Matching to AliceO2 Hits\n"
            << "  models_dir : " << models_dir << '\n'
            << "  n_events   : " << n_events << '\n'
            << "  output_dir : " << out_dir << '\n'
            << "  prefix     : " << prefix << '\n'
            << "  seed       : " << seed << '\n'
            << "  threads    : " << n_threads << "\n\n";

  // Download ONNX models if models_onnx.json specifies alien:// or ccdb:// paths
  try {
    download_models(models_dir.c_str());
  } catch (const std::exception& e) {
    LOG(fatal) << "Error downloading models: " << e.what();
    return 2;
  }

  // Load artefacts
  LOG(info) << "Loading artefacts ...\n";
  Artefacts art;
  try {
    art = load_artefacts(models_dir.c_str(), n_threads);
  } catch (const std::exception& e) {
    LOG(fatal) << "Error loading artefacts: " << e.what();
    return 2;
  }
  LOG(info) << "  Detectors : ";
  for (const auto& d : art.detectors)
    LOG(info) << d << ' ';
  LOG(info) << "\n  Patterns  : " << art.patterns.size()
            << "\n  CFM steps : " << art.cfm_steps << "\n\n";

  // Simulate
  std::mt19937 rng(seed);
  LOG(info) << "Simulating " << n_events << " events ...\n";
  auto t0 = std::chrono::steady_clock::now();
  auto events = simulate(n_events, art, rng);
  double elapsed =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  LOG(info) << "  Done in " << elapsed << " s  ("
            << 1000.0 * elapsed / n_events << " ms/event)\n";
  print_summary(events, art);

  // Write output
  LOG(info) << "\nWriting AliceO2 Hits.root files to " << out_dir << " ...\n";
  try {
    write_hits(events, art, out_dir.c_str(), prefix);
  } catch (const std::exception& e) {
    LOG(fatal) << "Error writing output: " << e.what();
    return 3;
  }

  LOG(info) << "  Written:\n";
  for (const auto& det : art.detectors)
    LOG(info) << "    " << out_dir << prefix << "_Hits" << det << ".root\n";

  return 0;
}
