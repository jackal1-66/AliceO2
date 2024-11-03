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
///
#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "ITStracking/Constants.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/TimeFrameGPU.h"
#include "ITStrackingGPU/TracerGPU.h"

#include <unistd.h>
#include <thread>

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonLogger.h"

#ifndef __HIPCC__
#define THRUST_NAMESPACE thrust::cuda
#else
#define THRUST_NAMESPACE thrust::hip
#endif

namespace o2
{
namespace its
{
using constants::GB;
using constants::MB;

namespace gpu
{
using utils::checkGPUError;

void* DefaultGPUAllocator::allocate(size_t size)
{
  LOGP(fatal, "Called DefaultGPUAllocator::allocate with size {}", size);
  return nullptr; // to be implemented
}

template <int nLayers>
TimeFrameGPU<nLayers>::TimeFrameGPU()
{
  mIsGPU = true;
  utils::getDeviceProp(0, true);
}

template <int nLayers>
TimeFrameGPU<nLayers>::~TimeFrameGPU() = default;

template <int nLayers>
void TimeFrameGPU<nLayers>::allocMemAsync(void** ptr, size_t size, Stream* strPtr, bool extAllocator)
{
  if (extAllocator) {
    *ptr = mAllocator->allocate(size);
  } else {
    LOGP(info, "Calling default CUDA allocator");
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(ptr), size, strPtr->get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::setDevicePropagator(const o2::base::PropagatorImpl<float>* propagator)
{
  mPropagatorDevice = propagator;
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadUnsortedClustersDevice()
{
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} unsorted clusters on layer {}, for {} MB.", mUnsortedClusters[iLayer].size(), iLayer, mUnsortedClusters[iLayer].size() * sizeof(Cluster) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mUnsortedClustersDevice[iLayer]), mUnsortedClusters[iLayer].size() * sizeof(Cluster), nullptr, getExtAllocator());
    // Register and move data
    checkGPUError(cudaHostRegister(mUnsortedClusters[iLayer].data(), mUnsortedClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mUnsortedClustersDevice[iLayer], mUnsortedClusters[iLayer].data(), mUnsortedClusters[iLayer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mUnsortedClustersDeviceArray), nLayers * sizeof(Cluster*), nullptr, getExtAllocator());
  checkGPUError(cudaHostRegister(mUnsortedClustersDevice.data(), nLayers * sizeof(Cluster*), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mUnsortedClustersDeviceArray, mUnsortedClustersDevice.data(), nLayers * sizeof(Cluster*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadClustersDevice()
{
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} clusters on layer {}, for {} MB.", mClusters[iLayer].size(), iLayer, mClusters[iLayer].size() * sizeof(Cluster) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mClustersDevice[iLayer]), mClusters[iLayer].size() * sizeof(Cluster), nullptr, getExtAllocator());
    // Register and move data
    checkGPUError(cudaHostRegister(mClusters[iLayer].data(), mClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mClustersDevice[iLayer], mClusters[iLayer].data(), mClusters[iLayer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mClustersDeviceArray), nLayers * sizeof(Cluster*), nullptr, getExtAllocator());
  checkGPUError(cudaHostRegister(mClustersDevice.data(), nLayers * sizeof(Cluster*), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mClustersDeviceArray, mClustersDevice.data(), nLayers * sizeof(Cluster*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackingFrameInfoDevice(const int iteration)
{
  if (!iteration) {
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      LOGP(debug, "gpu-transfer: loading {} tfinfo on layer {}, for {} MB.", mTrackingFrameInfo[iLayer].size(), iLayer, mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo) / MB);
      allocMemAsync(reinterpret_cast<void**>(&mTrackingFrameInfoDevice[iLayer]), mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), nullptr, getExtAllocator());
      // Register and move data
      checkGPUError(cudaHostRegister(mTrackingFrameInfo[iLayer].data(), mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaHostRegisterPortable));
      checkGPUError(cudaMemcpyAsync(mTrackingFrameInfoDevice[iLayer], mTrackingFrameInfo[iLayer].data(), mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
    }
    allocMemAsync(reinterpret_cast<void**>(&mTrackingFrameInfoDeviceArray), nLayers * sizeof(TrackingFrameInfo*), nullptr, getExtAllocator());
    checkGPUError(cudaHostRegister(mTrackingFrameInfoDevice.data(), nLayers * sizeof(TrackingFrameInfo*), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mTrackingFrameInfoDeviceArray, mTrackingFrameInfoDevice.data(), nLayers * sizeof(TrackingFrameInfo*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackletsDevice()
{
  for (auto iLayer{0}; iLayer < nLayers - 1; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} tracklets on layer {}, for {} MB.", mTracklets[iLayer].size(), iLayer, mTracklets[iLayer].size() * sizeof(Tracklet) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mTrackletsDevice[iLayer]), mTracklets[iLayer].size() * sizeof(Tracklet), nullptr, getExtAllocator());
    // Register and move data
    checkGPUError(cudaHostRegister(mTracklets[iLayer].data(), mTracklets[iLayer].size() * sizeof(Tracklet), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mTrackletsDevice[iLayer], mTracklets[iLayer].data(), mTracklets[iLayer].size() * sizeof(Tracklet), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mTrackletsDeviceArray), (nLayers - 1) * sizeof(Tracklet*), nullptr, getExtAllocator());
  checkGPUError(cudaHostRegister(mTrackletsDevice.data(), (nLayers - 1) * sizeof(Tracklet*), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mTrackletsDeviceArray, mTrackletsDevice.data(), (nLayers - 1) * sizeof(Tracklet*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadCellsDevice()
{
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} cell seeds on layer {}, for {} MB.", mCells[iLayer].size(), iLayer, mCells[iLayer].size() * sizeof(CellSeed) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mCellsDevice[iLayer]), mCells[iLayer].size() * sizeof(CellSeed), nullptr, getExtAllocator());
    allocMemAsync(reinterpret_cast<void**>(&mNeighboursIndexTablesDevice[iLayer]), (mCells[iLayer].size() + 1) * sizeof(int), nullptr, getExtAllocator()); // accessory for the neigh. finding.
    checkGPUError(cudaMemsetAsync(mNeighboursIndexTablesDevice[iLayer], 0, (mCells[iLayer].size() + 1) * sizeof(int), mGpuStreams[0].get()));
    // Register and move data
    checkGPUError(cudaHostRegister(mCells[iLayer].data(), mCells[iLayer].size() * sizeof(CellSeed), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mCellsDevice[iLayer], mCells[iLayer].data(), mCells[iLayer].size() * sizeof(CellSeed), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mCellsDeviceArray), (nLayers - 2) * sizeof(CellSeed*), nullptr, getExtAllocator());
  checkGPUError(cudaHostRegister(mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mCellsDeviceArray, mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadCellsLUT()
{
  for (auto iLayer{0}; iLayer < nLayers - 3; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} cell LUTs on layer {}, for {} MB.", mCellsLookupTable[iLayer].size(), iLayer, mCellsLookupTable[iLayer].size() * sizeof(int) / MB);
    allocMemAsync(reinterpret_cast<void**>(&(mCellsLUTDevice[iLayer])), sizeof(int) * mCellsLookupTable[iLayer].size(), nullptr, getExtAllocator());
    // Register and move data
    checkGPUError(cudaHostRegister(mCellsLookupTable[iLayer].data(), mCellsLookupTable[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mCellsLUTDevice[iLayer], mCellsLookupTable[iLayer].data(), mCellsLookupTable[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mCellsLUTDeviceArray), (nLayers - 2) * sizeof(int*), nullptr, getExtAllocator());
  checkGPUError(cudaHostRegister(mCellsLUTDevice.data(), mCellsLUTDevice.size() * sizeof(int*), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mCellsLUTDeviceArray, mCellsLUTDevice.data(), mCellsLUTDevice.size() * sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadRoadsDevice()
{
  LOGP(debug, "gpu-transfer: loading {} roads, for {} MB.", mRoads.size(), mRoads.size() * sizeof(Road<nLayers - 2>) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mRoadsDevice), mRoads.size() * sizeof(Road<nLayers - 2>), &(mGpuStreams[0]), getExtAllocator());
  checkGPUError(cudaHostRegister(mRoads.data(), mRoads.size() * sizeof(Road<nLayers - 2>), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mRoadsDevice, mRoads.data(), mRoads.size() * sizeof(Road<nLayers - 2>), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackSeedsDevice(std::vector<CellSeed>& seeds)
{
  LOGP(debug, "gpu-transfer: loading {} track seeds, for {} MB.", seeds.size(), seeds.size() * sizeof(CellSeed) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackSeedsDevice), seeds.size() * sizeof(CellSeed), &(mGpuStreams[0]), getExtAllocator());
  checkGPUError(cudaHostRegister(seeds.data(), seeds.size() * sizeof(CellSeed), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mTrackSeedsDevice, seeds.data(), seeds.size() * sizeof(CellSeed), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursDevice(const unsigned int& layer, std::vector<std::pair<int, int>>& neighbours)
{
  mCellsNeighbours[layer].clear();
  mCellsNeighbours[layer].resize(neighbours.size());
  LOGP(debug, "gpu-allocation: reserving {} neighbours, for {} MB.", neighbours.size(), neighbours.size() * sizeof(gpuPair<int, int>) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursDevice[layer]), neighbours.size() * sizeof(gpuPair<int, int>), &(mGpuStreams[0]), getExtAllocator());
  checkGPUError(cudaMemsetAsync(mNeighboursDevice[layer], -1, neighbours.size() * sizeof(gpuPair<int, int>), mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursLUTDevice(const int layer, const unsigned int nCells)
{
  LOGP(debug, "gpu-allocation: reserving {} slots for neighbours LUT, for {} MB.", nCells + 1, (nCells + 1) * sizeof(int) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursLUTDevice[layer]), (nCells + 1) * sizeof(int), nullptr, getExtAllocator()); // We need one element more to move exc -> inc
  checkGPUError(cudaMemsetAsync(mNeighboursLUTDevice[layer], 0, (nCells + 1) * sizeof(int), mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackITSExtDevice(std::vector<CellSeed>& seeds)
{
  mTrackITSExt.clear();
  mTrackITSExt.resize(seeds.size());
  LOGP(debug, "gpu-allocation: reserving {} tracks, for {} MB.", seeds.size(), seeds.size() * sizeof(o2::its::TrackITSExt) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackITSExtDevice), seeds.size() * sizeof(o2::its::TrackITSExt), &(mGpuStreams[0]), getExtAllocator());
  checkGPUError(cudaMemsetAsync(mTrackITSExtDevice, 0, seeds.size() * sizeof(o2::its::TrackITSExt), mGpuStreams[0].get()));
  checkGPUError(cudaHostRegister(mTrackITSExt.data(), seeds.size() * sizeof(o2::its::TrackITSExt), cudaHostRegisterPortable));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadCellsDevice(const int layer)
{
  LOGP(debug, "gpu-transfer: downloading {} cells on layer: {}, for {} MB.", mCells[layer].size(), layer, mCells[layer].size() * sizeof(CellSeed) / MB);
  checkGPUError(cudaMemcpyAsync(mCells[layer].data(), mCellsDevice[layer], mCells[layer].size() * sizeof(CellSeed), cudaMemcpyDeviceToHost, mGpuStreams[0].get()));
  checkGPUError(cudaHostUnregister(mCells[layer].data()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadCellsNeighbours(std::vector<std::vector<std::pair<int, int>>>& neighbours, const int layer)
{
  LOGP(debug, "gpu-transfer: downloading {} neighbours, for {} MB.", neighbours[layer].size(), neighbours[layer].size() * sizeof(std::pair<int, int>) / MB);
  // TOOD: something less dangerous than assuming the same memory layout of std::pair and gpuPair... or not? :)
  checkGPUError(cudaMemcpyAsync(neighbours[layer].data(), mNeighboursDevice[layer], neighbours[layer].size() * sizeof(gpuPair<int, int>), cudaMemcpyDeviceToHost, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadNeighboursLUT(std::vector<int>& lut, const int layer)
{
  LOGP(debug, "gpu-transfer: downloading {} neighbours lut, for {} MB.", lut.size(), lut.size() * sizeof(int) / MB);
  checkGPUError(cudaMemcpyAsync(lut.data(), mNeighboursLUTDevice[layer], lut.size() * sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadTrackITSExtDevice(std::vector<CellSeed>& seeds)
{
  LOGP(debug, "gpu-transfer: downloading {} tracks, for {} MB.", mTrackITSExt.size(), mTrackITSExt.size() * sizeof(o2::its::TrackITSExt) / MB);
  checkGPUError(cudaMemcpyAsync(mTrackITSExt.data(), mTrackITSExtDevice, seeds.size() * sizeof(o2::its::TrackITSExt), cudaMemcpyDeviceToHost, mGpuStreams[0].get()));
  checkGPUError(cudaHostUnregister(mTrackITSExt.data()));
  checkGPUError(cudaHostUnregister(seeds.data()));
  // discardResult(cudaDeviceSynchronize());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::unregisterRest()
{
  LOGP(debug, "unregistering rest of the host memory...");
  checkGPUError(cudaHostUnregister(mCells[0].data()));
  checkGPUError(cudaHostUnregister(mCellsDevice.data()));
  checkGPUError(cudaHostUnregister(mCellsLUTDevice.data()));
  for (auto iLayer{0}; iLayer < nLayers - 3; ++iLayer) {
    checkGPUError(cudaHostUnregister(mCellsLookupTable[iLayer].data()));
  }
}
////////////////////////////////////////////////////////////////////////
/// Legacy
template <int nLayers>
void TimeFrameGPU<nLayers>::registerHostMemory(const int maxLayers)
{
  if (mHostRegistered) {
    return;
  } else {
    mHostRegistered = true;
  }
  for (auto iLayer{0}; iLayer < maxLayers; ++iLayer) {
    checkGPUError(cudaHostRegister(mClusters[iLayer].data(), mClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
    checkGPUError(cudaHostRegister(mNClustersPerROF[iLayer].data(), mNClustersPerROF[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
    checkGPUError(cudaHostRegister(mIndexTables[iLayer].data(), (mStaticTrackingParams.ZBins * mStaticTrackingParams.PhiBins + 1) * mNrof * sizeof(int), cudaHostRegisterPortable));
  }
  checkGPUError(cudaHostRegister(mHostNTracklets.data(), (nLayers - 1) * mGpuParams.nTimeFrameChunks * sizeof(int), cudaHostRegisterPortable));
  checkGPUError(cudaHostRegister(mHostNCells.data(), (nLayers - 2) * mGpuParams.nTimeFrameChunks * sizeof(int), cudaHostRegisterPortable));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::unregisterHostMemory(const int maxLayers)
{
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    checkGPUError(cudaHostUnregister(mTrackingFrameInfo[iLayer].data()));
  }
  checkGPUError(cudaHostUnregister(mTrackingFrameInfoDevice.data()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initialise(const int iteration,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers,
                                       IndexTableUtils* utils,
                                       const TimeFrameGPUParameters* gpuParam)
{
  mGpuStreams.resize(mGpuParams.nTimeFrameChunks);
  o2::its::TimeFrame::initialise(iteration, trkParam, maxLayers);
}

template <int nLayers>
void TimeFrameGPU<nLayers>::wipe(const int maxLayers)
{
  unregisterHostMemory(maxLayers);
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initDevice(IndexTableUtils* utils,
                                       const TrackingParameters& trkParam,
                                       const TimeFrameGPUParameters& gpuParam,
                                       const int maxLayers,
                                       const int iteration)
{
  // mStaticTrackingParams.ZBins = trkParam.ZBins;
  // mStaticTrackingParams.PhiBins = trkParam.PhiBins;
  // if (mFirstInit) {
  //   mGpuParams = gpuParam;
  //   allocMemAsync(reinterpret_cast<void**>(&mTrackingParamsDevice), sizeof(gpu::StaticTrackingParameters<nLayers>), nullptr, true);
  //   checkGPUError(cudaMemcpy(mTrackingParamsDevice, &mStaticTrackingParams, sizeof(gpu::StaticTrackingParameters<nLayers>), cudaMemcpyHostToDevice));
  //   if (utils) { // If utils is not nullptr, then its gpu vertexing
  //     mIndexTableUtils = *utils;
  //     allocMemAsync(reinterpret_cast<void**>(&mIndexTableUtilsDevice), sizeof(IndexTableUtils), nullptr, true);
  //   } else { // GPU tracking otherwise
  //     mIndexTableUtils.setTrackingParameters(trkParam);
  //   }

  // mMemChunks.resize(mGpuParams.nTimeFrameChunks, GpuTimeFrameChunk<nLayers>{static_cast<TimeFrame*>(this), mGpuParams});
  // mVerticesInChunks.resize(mGpuParams.nTimeFrameChunks);
  // mNVerticesInChunks.resize(mGpuParams.nTimeFrameChunks);
  // mLabelsInChunks.resize(mGpuParams.nTimeFrameChunks);
  // LOGP(info, "Size of fixed part is: {} MB", GpuTimeFrameChunk<nLayers>::computeFixedSizeBytes(mGpuParams) / MB);
  // LOGP(info, "Size of scaling part is: {} MB", GpuTimeFrameChunk<nLayers>::computeScalingSizeBytes(GpuTimeFrameChunk<nLayers>::computeRofPerChunk(mGpuParams, mAvailMemGB), mGpuParams) / MB);
  // LOGP(info, "Allocating {} chunks of {} rofs capacity each.", mGpuParams.nTimeFrameChunks, mGpuParams.nROFsPerChunk);

  // for (int iChunk{0}; iChunk < mMemChunks.size(); ++iChunk) {
  //   mMemChunks[iChunk].allocate(GpuTimeFrameChunk<nLayers>::computeRofPerChunk(mGpuParams, mGpuParams.maxGPUMemoryGB), mGpuStreams[iChunk]);
  // }
  //   for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
  //     allocMemAsync(reinterpret_cast<void**>(&mROframesClustersDevice[iLayer]), mROframesClusters[iLayer].size() * sizeof(int), nullptr, true);
  //     allocMemAsync(reinterpret_cast<void**>(&(mUsedClustersDevice[iLayer])), sizeof(unsigned char) * mGpuParams.clustersPerROfCapacity * mNrof, nullptr, true);
  //   }
  //   allocMemAsync(reinterpret_cast<void**>(&mVerticesDevice), sizeof(Vertex) * mGpuParams.maxVerticesCapacity, nullptr, true);
  //   allocMemAsync(reinterpret_cast<void**>(&mROframesPVDevice), sizeof(int) * (mNrof + 1), nullptr, true);

  //   mFirstInit = false;
  // }
  // if (maxLayers < nLayers) { // Vertexer
  //   for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
  //     checkGPUError(cudaMemcpy(mROframesClustersDevice[iLayer], mROframesClusters[iLayer].data(), mROframesClusters[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice));
  //   }
  // } else { // Tracker
  //   checkGPUError(cudaMemcpy(mVerticesDevice, mPrimaryVertices.data(), sizeof(Vertex) * mPrimaryVertices.size(), cudaMemcpyHostToDevice));
  //   checkGPUError(cudaMemcpy(mROframesPVDevice, mROframesPV.data(), sizeof(int) * mROframesPV.size(), cudaMemcpyHostToDevice));
  //   if (!iteration) {
  //     for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
  //       checkGPUError(cudaMemset(mUsedClustersDevice[iLayer], 0, sizeof(unsigned char) * mGpuParams.clustersPerROfCapacity * mNrof));
  //     }
  //   }
  // }
  // checkGPUError(cudaMemcpy(mIndexTableUtilsDevice, &mIndexTableUtils, sizeof(IndexTableUtils), cudaMemcpyHostToDevice));
}

template <int nLayers>
unsigned char* TimeFrameGPU<nLayers>::getDeviceUsedClusters(const int layer)
{
  return mUsedClustersDevice[layer];
}

template <int nLayers>
gsl::span<int> TimeFrameGPU<nLayers>::getHostNTracklets(const int chunkId)
{
  return gsl::span<int>(mHostNTracklets.data() + (nLayers - 1) * chunkId, nLayers - 1);
}

template <int nLayers>
gsl::span<int> TimeFrameGPU<nLayers>::getHostNCells(const int chunkId)
{
  return gsl::span<int>(mHostNCells.data() + (nLayers - 2) * chunkId, nLayers - 2);
}

template class TimeFrameGPU<7>;
} // namespace gpu
} // namespace its
} // namespace o2
