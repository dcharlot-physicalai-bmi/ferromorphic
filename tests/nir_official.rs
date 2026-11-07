//! Canonical text produced from official neuromorphs/NIR HDF5 fixtures.
//!
//! `tools/nir_hdf5_oracle.py` reads the paper `.nir` files at commit
//! `563c58b45873429e3b1b6f71763ec334ceb981d5` and writes these files. This test
//! does not fetch them again. It checks that each file is a fixed point of the
//! text codec and that the bridge refuses the nodes it cannot represent.

use ferromorphic::nir::{BridgeError, Graph, Kind, Node, Rules, ValidationError};

fn load(name: &str) -> Graph {
    let path = format!("{}/tools/fixtures/nir/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{path}: {err}"));
    let graph = Graph::from_text(&text).unwrap_or_else(|err| panic!("{name}: {err}"));
    assert_eq!(graph.to_text(), text, "{name} is not canonical");
    graph
}

#[test]
fn official_nir_fixtures_are_canonical_text() {
    // Input -> weight -> LIF is legal NIR and outside the bridge. The bridge's input edge lands
    // on a LIF. These files put a weight node in between, so the refusal names that edge.
    for (file, from, to) in [
        ("lif_norse.nir.txt", "input", "0"),
        ("two_lif_neurons.nir.txt", "in", "linear1"),
    ] {
        let graph = load(file);
        graph.validate(&Rules::default()).unwrap_or_else(|err| panic!("{file}: {err}"));
        match graph.to_net(1e-4) {
            Err(BridgeError::UnsupportedEdge { from: got_from, to: got_to }) => {
                assert_eq!((got_from.as_str(), got_to.as_str()), (from, to), "{file}");
            }
            other => panic!("{file}: expected the input edge to be refused, got {other:?}"),
        }
    }

    let rockpool = load("lif_rockpool.nir.txt");
    let Node::Output(output) = &rockpool.nodes.iter().find(|n| n.name == "output").unwrap().node
    else {
        panic!("output");
    };
    assert_eq!(output.shape, vec![1, 1, 1], "the stored output shape is kept");
    match rockpool.validate(&Rules::default()) {
        Err(ValidationError::ShapeMismatch { from, to, emitted, expected }) => {
            assert_eq!(from, "1_LIFNeuronTorch");
            assert_eq!(to, "output");
            assert_eq!(emitted, vec![1]);
            assert_eq!(expected, vec![1, 1, 1]);
        }
        other => panic!("rockpool's stored shapes disagree, got {other:?}"),
    }

    let braille = load("braille_noDelay_bias_zero.nir.txt");
    braille.validate(&Rules::default()).expect("braille shapes agree");
    match braille.to_net(1e-4) {
        Err(BridgeError::UnsupportedNode { kind: Kind::CubaLif, .. }) => {}
        other => panic!("CubaLIF must be refused by name, got {other:?}"),
    }

    let cnn = load("cnn_sinabs.nir.txt");
    cnn.validate(&Rules::default()).expect("cnn shapes agree");
    let flat = cnn.nodes.iter().find(|n| n.name == "8").unwrap();
    let Node::Flatten(flat) = &flat.node else { panic!("node 8") };
    assert_eq!((flat.start_dim, flat.end_dim, flat.size.as_slice()), (0, 2, &[8, 4, 4][..]));
    match cnn.to_net(1e-4) {
        Err(BridgeError::UnsupportedNode { kind: Kind::Conv2d, name }) => {
            assert_eq!(name, "0");
        }
        other => panic!("the CNN is outside the bridge subset, got {other:?}"),
    }
}
