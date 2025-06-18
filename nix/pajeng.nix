{
  lib,
  fetchgit,
  stdenv,
  boost,
  cmake,
  flex,
  fmt,
  bison,
  asciidoc,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "pajeng";
  version = "master";

  src = fetchgit {
    url = "https://github.com/schnorr/pajeng.git";
    rev = "a702c38d19c0a2b97543f1bb16eb4485bc3b7291";
    hash = "sha256-2gOy/JGvpvdt8TelRVjRu+RwE0LRSdrK9cEeEX3tjiA=";
  };

  buildInputs = [
    boost
  ];

  nativeBuildInputs = [ 
    cmake
    flex
    fmt
    bison
    asciidoc
  ];

  cmakeFlags = [ "-DCMAKE_SKIP_BUILD_RPATH=ON" ];

  enableParallelBuilding = true;

  meta = {
    homepage = "https://github.com/schnorr/pajeng";
    description = "Tool for analysis of execution traces in the Paje File Format";

    longDescription = ''
      PajeNG (Paje Next Generation) is a re-implementation in C++ of the well-known Paje visualization tool for the analysis of execution traces (in the Paje File Format). The tool is released under the GNU General Public License 3. PajeNG comprises the libpaje library, and an auxiliary tool called pj_dump to transform Paje trace files to Comma-Separated Value (CSV). This dump tool also serves to validate the contents of a Paje trace file. The space-time visualization tool called pajeng had been deprecated (removed from the sources) since modern tools do a better job (see pj_gantt, for instance, or take a more general approach using R+ggplot2 to visualize the output of pj_dump). This effort was started as part of the french INFRA-SONGS ANR project. Development has continued through a collaboration between INF/UFRGS (Brazil) and Inria (France).
    '';
    license = lib.licenses.gpl3;
    maintainers = [ lib.maintainers.PhoqueEberlue ];
  };
})
