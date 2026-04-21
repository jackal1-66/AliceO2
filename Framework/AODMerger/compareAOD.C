// compareAOD.C
// Compare two AO2D.root files entry by entry.
// Checks: metaData, parentFiles keys, DF directory list, tree list, entry counts,
//         branch list, and every value of every branch for every entry.
//
// Usage:
//   root -l -b -q 'compareAOD.C("fileA.root","fileB.root")'
//
// Output per tree:
//   OK   DF_.../TreeName  (N entries, B branches)
//   FAIL DF_.../TreeName/branchName : M/N entries differ  (first few shown)

#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Compare two TTrees. Returns true if identical.
// ---------------------------------------------------------------------------
bool compareTrees(TTree* tA, TTree* tB, const std::string& path, int maxPrint = 5)
{
  Long64_t nA = tA->GetEntries();
  Long64_t nB = tB->GetEntries();
  bool ok = true;

  if (nA != nB) {
    printf("  FAIL %s: entry count %lld (A) vs %lld (B)\n", path.c_str(), nA, nB);
    return false;
  }

  // Build leaf pairs for branches present in both trees.
  struct Pair { TLeaf* lA; TLeaf* lB; std::string name; };
  std::vector<Pair> pairs;

  TObjArray* brsA = tA->GetListOfBranches();
  for (int i = 0; i < brsA->GetEntriesFast(); ++i) {
    const char* bname = brsA->UncheckedAt(i)->GetName();
    TBranch* brB = tB->GetBranch(bname);
    if (!brB) {
      printf("  FAIL %s: branch '%s' missing in B\n", path.c_str(), bname);
      ok = false;
      continue;
    }
    TLeaf* lA = (TLeaf*)tA->GetBranch(bname)->GetListOfLeaves()->First();
    TLeaf* lB = (TLeaf*)brB->GetListOfLeaves()->First();
    if (lA && lB) {
      pairs.push_back({lA, lB, bname});
    }
  }

  TObjArray* brsB = tB->GetListOfBranches();
  for (int i = 0; i < brsB->GetEntriesFast(); ++i) {
    const char* bname = brsB->UncheckedAt(i)->GetName();
    if (!tA->GetBranch(bname)) {
      printf("  FAIL %s: branch '%s' missing in A\n", path.c_str(), bname);
      ok = false;
    }
  }

  if (!ok) return false; // branch structure mismatch — skip value comparison

  // Per-branch difference counters.
  std::vector<long long> diffs(pairs.size(), 0);

  for (Long64_t ei = 0; ei < nA; ++ei) {
    tA->GetEntry(ei);
    tB->GetEntry(ei);

    for (size_t bi = 0; bi < pairs.size(); ++bi) {
      TLeaf* lA = pairs[bi].lA;
      TLeaf* lB = pairs[bi].lB;
      int lenA = lA->GetLen(); // actual length after GetEntry (handles VLA)
      int lenB = lB->GetLen();
      if (lenA != lenB) {
        if (diffs[bi] < maxPrint)
          printf("  DIFF %s/%s entry %lld: array length %d (A) vs %d (B)\n",
                 path.c_str(), pairs[bi].name.c_str(), ei, lenA, lenB);
        ++diffs[bi];
        continue;
      }
      for (int j = 0; j < lenA; ++j) {
        double vA = lA->GetValue(j);
        double vB = lB->GetValue(j);
        if (vA != vB) {
          if (diffs[bi] < maxPrint)
            printf("  DIFF %s/%s entry %lld [%d]: %.17g (A) vs %.17g (B)\n",
                   path.c_str(), pairs[bi].name.c_str(), ei, j, vA, vB);
          ++diffs[bi];
        }
      }
    }
  }

  for (size_t bi = 0; bi < pairs.size(); ++bi) {
    if (diffs[bi] > 0) {
      printf("  FAIL %s/%s: %lld/%lld values differ\n",
             path.c_str(), pairs[bi].name.c_str(), diffs[bi], nA);
      ok = false;
    }
  }

  if (ok)
    printf("  OK   %s  (%lld entries, %zu branches)\n",
           path.c_str(), nA, pairs.size());
  return ok;
}

// ---------------------------------------------------------------------------
// Compare two TMaps (for metaData / parentFiles).
// ---------------------------------------------------------------------------
bool compareMaps(TMap* mA, TMap* mB, const char* label)
{
  if (!mA && !mB) return true;
  if (!mA) { printf("  FAIL %s: missing in A\n", label); return false; }
  if (!mB) { printf("  FAIL %s: missing in B\n", label); return false; }

  bool ok = true;
  TIter next(mA);
  TObject* key;
  while ((key = next())) {
    TObjString* valA = (TObjString*)mA->GetValue(key);
    TObjString* valB = (TObjString*)mB->GetValue(key->GetName());
    if (!valB) {
      printf("  FAIL %s: key '%s' missing in B\n", label, key->GetName());
      ok = false;
    } else if (std::string(valA->GetName()) != std::string(valB->GetName())) {
      printf("  DIFF %s['%s']: '%s' (A) vs '%s' (B)\n",
             label, key->GetName(), valA->GetName(), valB->GetName());
      ok = false;
    }
  }
  TIter nextB(mB);
  while ((key = nextB())) {
    if (!mA->GetValue(key->GetName())) {
      printf("  FAIL %s: key '%s' missing in A\n", label, key->GetName());
      ok = false;
    }
  }
  if (ok) printf("  OK   %s\n", label);
  return ok;
}

// ---------------------------------------------------------------------------
void compareAOD(const char* fileA = "AO2D_serial.root",
                const char* fileB = "AO2D_parallel.root")
{
  auto fa = TFile::Open(fileA, "READ");
  auto fb = TFile::Open(fileB, "READ");
  if (!fa || fa->IsZombie()) { printf("ERROR: cannot open %s\n", fileA); return; }
  if (!fb || fb->IsZombie()) { printf("ERROR: cannot open %s\n", fileB); return; }

  printf("Comparing:\n  A: %s\n  B: %s\n\n", fileA, fileB);

  int nFail = 0;
  int nTrees = 0;

  // -- metaData --
  auto mA = (TMap*)fa->Get("metaData");
  auto mB = (TMap*)fb->Get("metaData");
  if (!compareMaps(mA, mB, "metaData")) ++nFail;

  // -- parentFiles --
  auto pA = (TMap*)fa->Get("parentFiles");
  auto pB = (TMap*)fb->Get("parentFiles");
  if (!compareMaps(pA, pB, "parentFiles")) ++nFail;

  printf("\n");

  // -- Collect DF directory names from both files --
  std::set<std::string> dfB;
  TList* keysB = fb->GetListOfKeys();
  for (TObject* o : *keysB) {
    std::string n = ((TKey*)o)->GetName();
    if (n.substr(0, 3) == "DF_") dfB.insert(n);
  }

  TList* keysA = fa->GetListOfKeys();
  keysA->Sort();
  std::set<std::string> dfA;

  for (TObject* o : *keysA) {
    TKey* key = (TKey*)o;
    std::string dfName = key->GetName();
    if (dfName.substr(0, 3) != "DF_") continue;
    if (dfA.count(dfName)) continue; // skip duplicate cycles
    dfA.insert(dfName);

    if (!dfB.count(dfName)) {
      printf("FAIL: DF '%s' missing in B\n", dfName.c_str());
      ++nFail;
      continue;
    }

    auto dirA = (TDirectoryFile*)fa->Get(dfName.c_str());
    auto dirB = (TDirectoryFile*)fb->Get(dfName.c_str());

    // Collect tree names from this DF in A (skip duplicate cycles)
    TList* treeKeysA = dirA->GetListOfKeys();
    treeKeysA->Sort();
    std::set<std::string> seenTrees;

    for (TObject* to : *treeKeysA) {
      std::string treeName = ((TKey*)to)->GetName();
      if (seenTrees.count(treeName)) continue;
      seenTrees.insert(treeName);

      auto tA = (TTree*)dirA->Get(treeName.c_str());
      auto tB = (TTree*)dirB->Get(treeName.c_str());
      if (!tB) {
        printf("  FAIL %s/%s: tree missing in B\n", dfName.c_str(), treeName.c_str());
        ++nFail;
        continue;
      }

      std::string path = dfName + "/" + treeName;
      if (!compareTrees(tA, tB, path)) ++nFail;
      ++nTrees;

      delete tA;
      delete tB;
    }

    // Check for trees only in B
    TList* treeKeysB = dirB->GetListOfKeys();
    std::set<std::string> seenB;
    for (TObject* to : *treeKeysB) {
      std::string treeName = ((TKey*)to)->GetName();
      if (seenB.count(treeName)) continue;
      seenB.insert(treeName);
      if (!seenTrees.count(treeName)) {
        printf("  FAIL %s/%s: tree missing in A\n", dfName.c_str(), treeName.c_str());
        ++nFail;
      }
    }
  }

  // Check for DF dirs only in B
  for (const auto& n : dfB) {
    if (!dfA.count(n)) {
      printf("FAIL: DF '%s' missing in A\n", n.c_str());
      ++nFail;
    }
  }

  printf("\n");
  if (nFail == 0)
    printf("RESULT: IDENTICAL  —  %d trees checked, no differences.\n", nTrees);
  else
    printf("RESULT: DIFFER  —  %d failure(s) across %d trees checked.\n", nFail, nTrees);

  fa->Close();
  fb->Close();
}
