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
#include <cstdlib>
#include <filesystem>
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
  std::vector<std::string> detectors;     // full ordered list from model_cfg.json (defines the context-vector layout)
  std::vector<std::string> sim_detectors; // subset selected via --detectors: only these are simulated and written
  int n_ctx{};
  int cfm_steps{};                   // integration steps read from model_cfg.json
  std::string cfm_method{"euler"};   // ODE solver: "euler" or "midpoint" (model_cfg.json)
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

// Detector-group aliases usable in --detectors
// "ALICE2" covers all the currently enabled detectors (Run 3).  
static const std::map<std::string, std::vector<std::string>> detector_groups = {
  {"ALICE2", {"FDD", "FT0", "FV0", "ITS", "MFT"}},
};

// "all" (or an empty string) applies no restriction to the subdetectors.
static std::vector<std::string> parse_detector_list(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  if (s.empty() || s == "ALL")
    return {};
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t comma = s.find(',', pos);
    std::string tok = (comma == std::string::npos) ? s.substr(pos) : s.substr(pos, comma - pos);
    if (!tok.empty()) {
      auto grp = detector_groups.find(tok);
      if (grp != detector_groups.end()) {
        for (const auto& det : grp->second)
          if (std::find(out.begin(), out.end(), det) == out.end())
            out.push_back(det);
      } else if (std::find(out.begin(), out.end(), tok) == out.end()) {
        out.push_back(tok);
      }
    }
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return out;
}

static Ort::Env& ort_env()
{
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "qedFastSim");
  return env;
}

// Directory where downloaded ONNX models are stored. When the production CCDB
// local cache is available (ALICEO2_CCDB_LOCALCACHE) the models go once into 
// <cache>/QEDFastGen/
static std::string onnx_cache_dir()
{
  const char* cache = std::getenv("ALICEO2_CCDB_LOCALCACHE");
  if (cache && *cache) {
    std::string dir = std::string(cache) + "/QEDFastGen/";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (!ec)
      return dir;
    LOG(warn) << "Cannot create " << dir << " (" << ec.message() << "), ONNX models will go to ./";
  }
  return "./";
}

// -- Download ONNX models listed in models_onnx.json
// The JSON maps detector names to their source paths, e.g.:
//   { "FDD": "alien://path/to/cfm_FDD.onnx", "FT0": "ccdb://Users/.../cfm_FT0", ... }
// Each model is downloaded into <onnx_dir>/cfm_<DET>.onnx, which is the path
// that load_artefacts() looks up first.  Local paths are ignored (nothing to
// download). Only the detectors in `selected` are fetched (empty = all).
// Models already present in onnx_dir are reused.  Downloading is meant to run
// as a dedicated single step (--download-models) before any parallel
// simulation tasks start.
static void download_models(const std::string& dir, const std::vector<std::string>& selected,
                            const std::string& onnx_dir)
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
    if (!selected.empty() &&
        std::find(selected.begin(), selected.end(), det) == selected.end())
      continue; // detector not selected: skip download
    const std::string src = val.get<std::string>();
    const std::string local = onnx_dir + "cfm_" + det + ".onnx";
    std::error_code ec;
    if (std::filesystem::exists(local, ec) && std::filesystem::file_size(local, ec) > 0) {
      LOG(info) << "  [cache] " << local << " already present, skipping download\n";
      continue;
    }
    if (src.starts_with("alien://"))
      alien_entries.push_back({det, src, local});
    else if (src.starts_with("ccdb://"))
      ccdb_entries.push_back({det, src, local});
    // else: already a local path, nothing to download
  }

  if (alien_entries.empty() && ccdb_entries.empty())
    return;

  LOG(info) << "Downloading ONNX models to " << onnx_dir << " ...\n";

  // interrupted downloads leave only a .part file, never a truncated final file
  const std::string tmp_suffix = ".part";
  auto commit = [](const std::string& tmp, const std::string& local) {
    std::error_code ec;
    std::filesystem::rename(tmp, local, ec);
    if (ec) {
      LOG(fatal) << "Error: Cannot move " << tmp << " to " << local << ": " << ec.message();
      exit(1);
    }
  };

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
      const std::string tmp = e.local + tmp_suffix;
      if (!TFile::Cp(e.src.c_str(), tmp.c_str())) {
        LOG(fatal) << "Error: Model file " << e.src << " does not exist or failed to copy!";
        exit(1);
      }
      commit(tmp, e.local);
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
      const std::string tmp_fname = "cfm_" + e.det + ".onnx" + tmp_suffix;
      if (!api.retrieveBlob(ccdb_path, onnx_dir, filter, ts, false, tmp_fname.c_str())) {
        LOG(fatal) << "Error: Failed to retrieve " << ccdb_path << " from CCDB!";
        exit(1);
      }
      commit(onnx_dir + tmp_fname, e.local);
    }
  }
}

// -- Load artefacts
// `selected` restricts the simulated detectors (empty = all detectors in
// model_cfg.json).
static Artefacts load_artefacts(const std::string& dir, int n_threads,
                                const std::vector<std::string>& selected,
                                const std::string& onnx_dir)
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
    // Configs written before 2026-07 carry no CFM_METHOD key -> default "euler".
    art.cfm_steps = j.value("CFM_STEPS", 10);
    art.cfm_method = j.value("CFM_METHOD", std::string("euler"));
    if (art.cfm_method != "euler" && art.cfm_method != "midpoint") {
      LOG(fatal) << "Unknown sampler method in model_cfg.json: " << art.cfm_method;
      exit(1);
    }
    if (selected.empty()) {
      art.sim_detectors = art.detectors;
    } else {
      for (const auto& det : selected)
        if (std::find(art.detectors.begin(), art.detectors.end(), det) == art.detectors.end()) {
          std::string avail;
          for (const auto& d : art.detectors)
            avail += d + " ";
          LOG(error) << "Unknown detector '" << det << "' in --detectors (available: " << avail << ")";
          exit(1);
        }
      // keep config order so loops stay deterministic
      for (const auto& det : art.detectors)
        if (std::find(selected.begin(), selected.end(), det) != selected.end())
          art.sim_detectors.push_back(det);
    }
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
  // Models are looked up in the download/cache dir first (where download_models()
  // puts them), then in the artefact dir (fully local setups), then in the cwd
  // (legacy behaviour). Only the selected detectors need a session.
  for (const auto& det : art.sim_detectors) {
    const std::string fname = "cfm_" + det + ".onnx";
    std::string path = onnx_dir + fname;
    if (!std::filesystem::exists(path)) {
      if (std::filesystem::exists(dir + fname))
        path = dir + fname;
      else if (std::filesystem::exists(fname))
        path = fname;
      else {
        LOG(fatal) << "Error: Cannot find " << fname << " in " << onnx_dir << ", " << dir << " or ./";
        exit(1);
      }
    }
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
  const auto& dets = art.detectors;         // full list: defines the context-vector layout
  const auto& sim_dets = art.sim_detectors; // only these are sampled and returned

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

    for (const auto& det : sim_dets) {
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

  // Stage 3: ODE integration from x_0 ~ N(0,I) at t=0 to the data sample at t=1,
  // with dt = 1/cfm_steps and the velocity MLP (x_t, t_vec, ctx) -> velocity:
  //   euler    (1 ONNX call/step):  x += dt * v(x, t)
  //   midpoint (2 ONNX calls/step): k1 = v(x, t);  x += dt * v(x + dt/2*k1, t + dt/2)
  // The midpoint (RK2) rule is the notebook default since 2026-07: at the same
  // total number of MLP calls it removes almost all integration error
  // (measured KS mean 0.0151 for midpoint-5 vs 0.0347 for euler-10).
  const float dt = 1.0f / static_cast<float>(art.cfm_steps);
  const bool use_midpoint = (art.cfm_method == "midpoint");

  // Normal distribution provides the starting noise for the features
  std::normal_distribution<float> normal(0.f, 1.f);
  std::map<std::string, std::map<int, std::vector<std::vector<float>>>> all_hits;
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  for (const auto& det : sim_dets) {
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

    // One velocity evaluation: returns Run() output holding v(x_in, t_scalar, ctx).
    auto eval_velocity = [&](std::vector<float>& x_in, float t_scalar) {
      std::fill(t_vec.begin(), t_vec.end(), t_scalar);

      Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
        mem_info, x_in.data(), static_cast<size_t>(total * n_feat),
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

      return os.session->Run(
        Ort::RunOptions{nullptr},
        os.input_names.data(), inputs.data(), inputs.size(),
        os.output_names.data(), os.output_names.size());
    };

    const int sz = total * n_feat;
    std::vector<float> x_half;
    if (use_midpoint)
      x_half.resize(sz);

    for (int step = 0; step < art.cfm_steps; ++step) {
      const float t0 = step * dt;
      if (use_midpoint) {
        // k1 = v(x, t);  x += dt * v(x + dt/2*k1, t + dt/2)
        auto out1 = eval_velocity(x_data, t0);
        const float* k1 = out1[0].GetTensorData<float>();
        for (int k = 0; k < sz; ++k)
          x_half[k] = x_data[k] + 0.5f * dt * k1[k];
        auto out2 = eval_velocity(x_half, t0 + 0.5f * dt);
        const float* v2 = out2[0].GetTensorData<float>();
        for (int k = 0; k < sz; ++k)
          x_data[k] += dt * v2[k];
      } else {
        // x += dt * v(x, t)
        auto out = eval_velocity(x_data, t0);
        const float* vel = out[0].GetTensorData<float>();
        for (int k = 0; k < sz; ++k)
          x_data[k] += vel[k] * dt;
      }
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
    for (const auto& det : sim_dets)
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

// -- Incremental per-detector Hits.root writer
// The files/trees are opened once, then fill() is called per simulated chunk of
// events and streams the hits to disk immediately.  This bounds peak memory to
// one chunk instead of the full generation (the previous version held all
// events in RAM before writing, which exhausted memory around 180k events).
struct HitWriter {
  std::vector<o2::fdd::Hit> fdd_hits;
  std::vector<o2::ft0::HitType> ft0_hits;
  std::vector<o2::fv0::Hit> fv0_hits;
  std::vector<o2::itsmft::Hit> its_hits;
  std::vector<o2::itsmft::Hit> mft_hits;

  TFile *fdd_file{nullptr}, *ft0_file{nullptr}, *fv0_file{nullptr},
        *its_file{nullptr}, *mft_file{nullptr};
  TTree *fdd_tree{nullptr}, *ft0_tree{nullptr}, *fv0_tree{nullptr},
        *its_tree{nullptr}, *mft_tree{nullptr};
  bool has_fdd{}, has_ft0{}, has_fv0{}, has_its{}, has_mft{};

  // Accumulated statistics across all chunks (for the final summary)
  std::map<std::string, int> n_active, n_hits;
  long n_events_total{0};

  void open(const Artefacts& art, const std::string& out_dir, const std::string& prefix)
  {
    auto open_file = [&](const std::string& fname) -> std::pair<TFile*, TTree*> {
      TFile* f = TFile::Open(fname.c_str(), "RECREATE");
      if (!f || f->IsZombie()) {
        LOG(fatal) << "Error: Cannot create file " << fname;
        exit(1);
      }
      auto* t = new TTree("o2sim", "o2sim");
      return {f, t};
    };

    has_fdd = std::count(art.sim_detectors.begin(), art.sim_detectors.end(), "FDD");
    has_ft0 = std::count(art.sim_detectors.begin(), art.sim_detectors.end(), "FT0");
    has_fv0 = std::count(art.sim_detectors.begin(), art.sim_detectors.end(), "FV0");
    has_its = std::count(art.sim_detectors.begin(), art.sim_detectors.end(), "ITS");
    has_mft = std::count(art.sim_detectors.begin(), art.sim_detectors.end(), "MFT");

    if (has_fdd) {
      std::tie(fdd_file, fdd_tree) = open_file(out_dir + prefix + "_HitsFDD.root");
      fdd_tree->Branch("FDDHit", &fdd_hits);
    }
    if (has_ft0) {
      std::tie(ft0_file, ft0_tree) = open_file(out_dir + prefix + "_HitsFT0.root");
      ft0_tree->Branch("FT0Hit", &ft0_hits);
    }
    if (has_fv0) {
      std::tie(fv0_file, fv0_tree) = open_file(out_dir + prefix + "_HitsFV0.root");
      fv0_tree->Branch("FV0Hit", &fv0_hits);
    }
    if (has_its) {
      std::tie(its_file, its_tree) = open_file(out_dir + prefix + "_HitsITS.root");
      its_tree->Branch("ITSHit", &its_hits);
    }
    if (has_mft) {
      std::tie(mft_file, mft_tree) = open_file(out_dir + prefix + "_HitsMFT.root");
      mft_tree->Branch("MFTHit", &mft_hits);
    }
  }

  void fill(const std::vector<Event>& events)
  {
    for (const auto& event : events) {
      const int trackID = 0;

      if (has_fdd) {
        fdd_hits.clear();
        auto it = event.find("FDD");
        if (it != event.end()) {
          for (const auto& feat : it->second)
            fdd_hits.push_back(make_fdd_hit(trackID, feat));
          if (!it->second.empty()) {
            ++n_active["FDD"];
            n_hits["FDD"] += static_cast<int>(it->second.size());
          }
        }
        fdd_file->cd();
        fdd_tree->Fill();
      }

      if (has_ft0) {
        ft0_hits.clear();
        auto it = event.find("FT0");
        if (it != event.end()) {
          for (const auto& feat : it->second)
            ft0_hits.push_back(make_ft0_hit(trackID, feat));
          if (!it->second.empty()) {
            ++n_active["FT0"];
            n_hits["FT0"] += static_cast<int>(it->second.size());
          }
        }
        ft0_file->cd();
        ft0_tree->Fill();
      }

      if (has_fv0) {
        fv0_hits.clear();
        auto it = event.find("FV0");
        if (it != event.end()) {
          for (const auto& feat : it->second)
            fv0_hits.push_back(make_fv0_hit(trackID, feat));
          if (!it->second.empty()) {
            ++n_active["FV0"];
            n_hits["FV0"] += static_cast<int>(it->second.size());
          }
        }
        fv0_file->cd();
        fv0_tree->Fill();
      }

      if (has_its) {
        its_hits.clear();
        auto it = event.find("ITS");
        if (it != event.end()) {
          for (const auto& feat : it->second)
            its_hits.push_back(make_itsmft_hit(trackID, feat));
          if (!it->second.empty()) {
            ++n_active["ITS"];
            n_hits["ITS"] += static_cast<int>(it->second.size());
          }
        }
        its_file->cd();
        its_tree->Fill();
      }

      if (has_mft) {
        mft_hits.clear();
        auto it = event.find("MFT");
        if (it != event.end()) {
          for (const auto& feat : it->second)
            mft_hits.push_back(make_itsmft_hit(trackID, feat));
          if (!it->second.empty()) {
            ++n_active["MFT"];
            n_hits["MFT"] += static_cast<int>(it->second.size());
          }
        }
        mft_file->cd();
        mft_tree->Fill();
      }
    }
    n_events_total += static_cast<long>(events.size());
  }

  void close()
  {
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
    fdd_file = ft0_file = fv0_file = its_file = mft_file = nullptr;
  }

  void print_summary(const Artefacts& art) const
  {
    LOG(info) << "\n  Per-detector statistics:\n";
    for (const auto& det : art.sim_detectors) {
      LOG(info) << "    " << det
                << "  active_events=" << (n_active.count(det) ? n_active.at(det) : 0)
                << "/" << n_events_total
                << "  total_hits=" << (n_hits.count(det) ? n_hits.at(det) : 0) << "\n";
    }
  }
};

// -- Entry point ---------------------------------------------------------------
namespace bpo = boost::program_options;

int main(int argc, char* argv[])
{
  // Single-thread by default: ONNX inference is fast enough per timeframe and
  // a 1-core task schedules better in full O2DPG workflows; use -j to raise it.
  const int default_threads = 1;

  bpo::options_description options("qedFastSimCFM options");
  options.add_options()
    ("models-dir,m", bpo::value<std::string>()->default_value("${O2_ROOT}/share/Generators/QEDFastGen"),
                                                                        "directory with CFM artefacts from fastsim_cfm.ipynb")
    ("nevents,n",    bpo::value<int>(),                                 "number of events to simulate (required unless --download-models)")
    ("output-dir,o", bpo::value<std::string>()->default_value("./"),    "directory where Hits.root files are written")
    ("prefix,p",     bpo::value<std::string>()->default_value("qed"), "file-name prefix")
    ("seed,s",       bpo::value<uint32_t>()->default_value(42u),        "RNG seed")
    ("threads,j",    bpo::value<int>()->default_value(default_threads), "ORT intra/inter-op thread count (default 1 = single-thread)")
    ("chunk,c",      bpo::value<int>()->default_value(20000),           "events simulated and written per chunk (bounds peak memory)")
    ("detectors,d",  bpo::value<std::string>()->default_value("ALICE2"),   "comma-separated detectors to simulate (e.g. FT0,ITS), a group alias (ALICE2), or 'all'")
    ("download-models", "download the ONNX models into the cache directory and exit; run this once before starting (parallel) simulation tasks")
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

  const bool download_only = vm.count("download-models") > 0;
  if (!download_only && !vm.count("nevents")) {
    LOG(error) << "Missing required option --nevents (-n)\n\n";
    std::cerr << options;
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
  const int n_events            = download_only ? 0 : vm["nevents"].as<int>();
  const std::string out_dir    = expand(ensure_slash(vm["output-dir"].as<std::string>()));
  const std::string prefix      = vm["prefix"].as<std::string>();
  const uint32_t seed           = vm["seed"].as<uint32_t>();
  const int n_threads           = std::max(1, vm["threads"].as<int>());
  const int chunk_size          = std::max(1, vm["chunk"].as<int>());
  const auto selected_dets      = parse_detector_list(vm["detectors"].as<std::string>());
  const std::string onnx_dir    = onnx_cache_dir();

  LOG(info) << "qedFastSimCFM - QED Conditional Flow Matching to AliceO2 Hits\n"
            << "  models_dir : " << models_dir << '\n'
            << "  onnx_cache : " << onnx_dir << '\n'
            << "  n_events   : " << n_events << '\n'
            << "  output_dir : " << out_dir << '\n'
            << "  prefix     : " << prefix << '\n'
            << "  seed       : " << seed << '\n'
            << "  threads    : " << n_threads << '\n'
            << "  chunk      : " << chunk_size << '\n'
            << "  detectors  : " << vm["detectors"].as<std::string>() << "\n\n";

  // Download ONNX models if models_onnx.json specifies alien:// or ccdb:// paths
  try {
    download_models(models_dir.c_str(), selected_dets, onnx_dir);
  } catch (const std::exception& e) {
    LOG(fatal) << "Error downloading models: " << e.what();
    return 2;
  }

  if (download_only) {
    // dedicated download step (e.g. the O2DPG qedmodeldownload task)
    LOG(info) << "Models available in " << onnx_dir << ":\n";
    if (selected_dets.empty()) {
      LOG(info) << "  (all detectors listed in models_onnx.json)\n";
    } else {
      for (const auto& det : selected_dets)
        LOG(info) << "  cfm_" << det << ".onnx\n";
    }
    return 0;
  }

  // Load artefacts
  LOG(info) << "Loading artefacts ...\n";
  Artefacts art;
  try {
    art = load_artefacts(models_dir.c_str(), n_threads, selected_dets, onnx_dir);
  } catch (const std::exception& e) {
    LOG(fatal) << "Error loading artefacts: " << e.what();
    return 2;
  }
  LOG(info) << "  Detectors : ";
  for (const auto& d : art.sim_detectors)
    LOG(info) << d << ' ';
  LOG(info) << "\n  Patterns  : " << art.patterns.size()
            << "\n  Sampler   : " << art.cfm_method << "-" << art.cfm_steps
            << "  (" << art.cfm_steps * (art.cfm_method == "midpoint" ? 2 : 1)
            << " MLP calls/sample)\n\n";

  // Simulate + write in chunks: each chunk is generated, streamed to the ROOT
  // trees, and freed before the next one starts, so peak memory is bounded by
  // chunk_size events regardless of the total generation size.
  std::mt19937 rng(seed);
  HitWriter writer;
  try {
    writer.open(art, out_dir.c_str(), prefix);
  } catch (const std::exception& e) {
    LOG(fatal) << "Error opening output files: " << e.what();
    return 3;
  }

  LOG(info) << "Simulating " << n_events << " events (chunks of " << chunk_size << ") ...\n";
  auto t0 = std::chrono::steady_clock::now();
  try {
    for (int done = 0; done < n_events; done += chunk_size) {
      const int n_chunk = std::min(chunk_size, n_events - done);
      auto events = simulate(n_chunk, art, rng);
      writer.fill(events);
      LOG(info) << "  " << (done + n_chunk) << " / " << n_events << " events written\n";
    }
    writer.close();
  } catch (const std::exception& e) {
    LOG(fatal) << "Error during simulation/writing: " << e.what();
    return 3;
  }
  double elapsed =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  LOG(info) << "  Done in " << elapsed << " s  ("
            << 1000.0 * elapsed / n_events << " ms/event)\n";
  writer.print_summary(art);

  LOG(info) << "  Written:\n";
  for (const auto& det : art.sim_detectors)
    LOG(info) << "    " << out_dir << prefix << "_Hits" << det << ".root\n";

  return 0;
}
