{
  stdenv,
  fetchFromGitLab,
  lib,
  cmake,
  qtPackages,
  libGLU,
  libGL,
  glm,
  glew,
}:

stdenv.mkDerivation rec {
  pname = "vite";
  version = "1.4";

  src = fetchFromGitLab {
    domain = "gitlab.inria.fr";
    owner = "solverstack";
    repo = pname;
    rev = "v${version}";
    hash = "sha256-z2M4BazLzO/NCcq/VKb+tgrZ6QUs+AX0BbzJW809Krg=";
  };

  nativeBuildInputs = [
    cmake
    qtPackages.qttools
    qtPackages.wrapQtAppsHook
  ];

  buildInputs = [
    qtPackages.qtbase
    qtPackages.qtcharts
    libGLU
    libGL
    glm
    glew
  ];

  meta = {
    description = "Visual Trace Explorer (ViTE), a tool to visualize execution traces";
    mainProgram = "vite";

    longDescription = ''
      ViTE is a trace explorer. It is a tool to visualize execution
      traces in Pajé or OTF format for debugging and profiling
      parallel or distributed applications.
    '';

    homepage = "http://vite.gforge.inria.fr/";
    license = lib.licenses.cecill20;
    maintainers = [ ];
    platforms = lib.platforms.linux;
  };
}
