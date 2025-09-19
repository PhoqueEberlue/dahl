{
  lib,
  fetchFromGitHub,
  rustPlatform,
  openssl,
  pkg-config
}:
rustPlatform.buildRustPackage rec {
  pname = "alumet";
  version = "0.9.0";

  src = fetchFromGitHub {
    owner = "alumet-dev";
    repo = pname;
    rev = "v${version}";
    hash = "sha256-G1/BqJoj00RDFB8w9r+ZD/nEuhJ0xDSupoY86+T+25k=";
  };

  buildInputs = [ openssl pkg-config ];
  nativeBuildInputs = [ openssl pkg-config ];

  cargoHash = "sha256-hQTGTlv3TuMi36bJKh6vO5eE27a6+hjat9EYWRi2rbc=";
  doCheck = false;
  enableParallelBuilding = true;

  meta = with lib; {
    description = "Unified and Modular Measurement Framework";
    homepage = "https://alumet.dev/";
    license = licenses.eupl12;
    maintainers = [ maintainers.PhoqueEberlue ];
  };
}
