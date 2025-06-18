{ lib, fetchzip, stdenv, perl }:

stdenv.mkDerivation (finalAttrs: {
  pname = "fxt";
  version = "0.3.15";

  src = fetchzip {
    url = "https://download.savannah.nongnu.org/releases/fkt/fxt-${finalAttrs.version}.tar.gz";
    hash = "sha256-5sBbzF2ua7t9wjh4L5B/Gzjp2AVDdxmpZ6WugQXBxrI=";
  };

  nativeBuildInputs = [ perl ];

  enableParallelBuilding = true;
  doCheck = true;

  meta = {
    homepage = "https://savannah.nongnu.org/projects/fkt/";
    description = "This library provides efficient support for recording User/Kernel traces.";
    longDescription = ''
      FxT stands for both FKT (Fast Kernel Tracing) and FUT (Fast User Tracing). This library provides efficient support for recording traces.
    '';
    license = lib.licenses.gpl2;
    maintainers = [ lib.maintainers.PhoqueEberlue ];
  };
})
