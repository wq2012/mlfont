"""Comprehensive unit tests for the MLFont open-source package."""

from __future__ import annotations

from pathlib import Path
import random

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
import numpy as np

from mlfont import (
    DecayConfig,
    DnnTmConfig,
    DnnTmLearner,
    DocumentCorpusProcessor,
    EvaluationParams,
    ExponentialDecay,
    FontLoadingEvaluator,
    OnlineLdaConfig,
    OnlineLdaLearner,
    PowerDecay,
    RangeCompactConfig,
    RangeCompactGeneticLearner,
    SlicingStrategy,
    balance_topics_diff_constrained,
    balance_topics_greedy,
    decrease_max_topic_mutation,
    export_pretrained_bundle,
    extract_codepoints_from_font,
    fill_codepoint_gaps,
    format_unicode_ranges,
    generate_equal_bin_strategy,
    generate_language_slice_strategy,
    generate_min_slice_strategy,
    generate_most_and_equal_bin_strategy,
    generate_random_slice_strategy,
    generate_single_slice_strategy,
    get_equally_sized_chunks,
    get_preset_config,
    get_slices_based_on_frequency,
    increase_min_topic_mutation,
    init_gene_by_equal_binning,
    init_gene_from_strategy,
    init_gene_from_topic_matrix,
    left_shift_mutation,
    load_pretrained,
    merge_partial_strategies,
    merge_short_range_mutation,
    normalize_lambda_matrix,
    parse_codepoints_file,
    preprocess_codepoint_list_and_lambda_matrix,
    purely_random_crossover,
    purely_random_mutation,
    right_shift_mutation,
    save_codepoints_file,
    save_evaluation_results_json,
    slice_font_file,
    swap_topic_mutation,
)


def _create_tiny_ttf(path: Path, codepoints: list[int]) -> Path:
    """Builds a minimal valid TrueType font supporting the given codepoints."""
    fb = FontBuilder(1000, isTTF=True)
    glyph_order = [".notdef"] + [f"uni{cp:04X}" for cp in codepoints]
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap({cp: f"uni{cp:04X}" for cp in codepoints})

    pen = TTGlyphPen(None)
    pen.moveTo((100, 100))
    pen.lineTo((100, 700))
    pen.lineTo((600, 700))
    pen.lineTo((600, 100))
    pen.closePath()
    glyph = pen.glyph()

    glyphs = {gname: glyph for gname in glyph_order}
    fb.setupGlyf(glyphs)
    metrics = {gname: (600, 100) for gname in glyph_order}
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TinyTest", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    fb.save(str(path))
    return path


def test_slicing_strategy_dna_and_serialization(tmp_path: Path) -> None:
    strat = SlicingStrategy(
        subsets=[[0x41, 0x42, 0x43], [0x44, 0x46], [0x45]],
        name="test_strat",
    )
    assert strat.num_slices == 3
    assert strat.num_codepoints == 6
    assert strat.count_unicode_ranges() == 4
    assert strat.count_dna_compactness_ranges() == 4

    dna = strat.to_dna()
    reconstructed = SlicingStrategy.from_dna(dna, num_topics=3, name="reconstructed")
    assert reconstructed.subsets == strat.subsets

    json_path = tmp_path / "strat.json"
    strat.save_json(json_path)
    loaded = SlicingStrategy.load_json(json_path)
    assert loaded.subsets == strat.subsets
    assert loaded.name == "test_strat"

    nam_files = strat.save_nam_dir(tmp_path / "nams")
    assert len(nam_files) == 3
    assert sorted(parse_codepoints_file(nam_files[0])) == [0x41, 0x42, 0x43]


def test_data_parsing_and_corpus_processor(tmp_path: Path) -> None:
    nam_file = tmp_path / "sample.nam"
    nam_file.write_text(
        "# Comment line\n0x0041 A\n0x0042 B\nU+0044 D\n0x0045 E\n",
        encoding="utf-8",
    )
    cps = parse_codepoints_file(nam_file)
    assert cps == {0x41, 0x42, 0x44, 0x45}
    filled = fill_codepoint_gaps(cps, fill_gap=2)
    assert filled == {0x41, 0x42, 0x43, 0x44, 0x45}

    out_nam = tmp_path / "saved.nam"
    save_codepoints_file(filled, out_nam)
    assert parse_codepoints_file(out_nam) == filled

    docs = ["AB", "ABC", "AD", "BCD"]
    proc = DocumentCorpusProcessor(docs, random_seed=0)
    valid_cps = {0x41, 0x42, 0x43, 0x44}
    freqs = proc.count_codepoint_frequency(valid_cps, deduplicate_within_doc=True)
    assert freqs[0x41] == 3  # in "AB", "ABC", "AD"
    assert freqs[0x42] == 3  # in "AB", "ABC", "BCD"
    assert freqs[0x43] == 2  # in "ABC", "BCD"
    assert freqs[0x44] == 2  # in "AD", "BCD"

    batch = proc.construct_batch_sparse_matrix([0x41, 0x42, 0x43, 0x44], n_samples=3)
    assert batch.shape == (3, 4)


def test_normalize_and_topic_balancing() -> None:
    mat = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 2.0, 1.0, 0.0]], dtype=np.float64)
    col_norm = normalize_lambda_matrix(mat, "topic_sum")
    np.testing.assert_allclose(np.sum(col_norm, axis=0), np.ones(4))

    row_norm = normalize_lambda_matrix(mat, "row_sum")
    np.testing.assert_allclose(np.sum(row_norm, axis=1), np.ones(2))

    max_norm = normalize_lambda_matrix(mat, "global_max")
    assert np.isclose(np.max(max_norm), 1.0)

    spec_norm = normalize_lambda_matrix(mat, "spectral_norm")
    assert np.isclose(np.linalg.norm(spec_norm, ord=2), 1.0)

    sink_norm = normalize_lambda_matrix(mat, "sinkhorn")
    assert sink_norm.shape == (2, 4)
    np.testing.assert_allclose(np.sum(sink_norm, axis=0), np.ones(4), atol=1e-2)
    np.testing.assert_allclose(np.sum(sink_norm, axis=1), np.array([2.0, 2.0]), atol=1e-2)

    greedy = balance_topics_greedy(mat, "topic_sum")
    assert greedy.shape == (2, 4)
    np.testing.assert_array_equal(np.sum(greedy, axis=0), np.ones(4, dtype=np.uint8))
    np.testing.assert_array_equal(np.sum(greedy, axis=1), np.array([2, 2]))

    diff_c = balance_topics_diff_constrained(mat, max_diff=1, normalization_method="topic_sum")
    assert diff_c.shape == (2, 4)
    np.testing.assert_array_equal(np.sum(diff_c, axis=0), np.ones(4, dtype=np.uint8))


def test_parameter_decay() -> None:
    cfg = DecayConfig(min_val=10.0, max_val=50.0, decay_method="POWER", decay_power=3.0)
    p_decay = PowerDecay(cfg, max_step=100)
    assert np.isclose(p_decay(0), 50.0)
    assert np.isclose(p_decay(100), 10.0)
    assert 10.0 < p_decay(50) < 50.0

    e_cfg = DecayConfig(min_val=10.0, max_val=50.0, decay_method="EXPONENTIAL", decay_rate=0.1)
    e_decay = ExponentialDecay(e_cfg, max_step=100)
    assert np.isclose(e_decay(0), 50.0)
    assert np.isclose(e_decay(100), 10.0)


def test_frequency_slicing() -> None:
    chunks = get_equally_sized_chunks([1, 2, 3, 4, 5], 2)
    assert chunks == [[1, 2, 3], [4, 5]]

    cps = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E")]
    freq_arr = np.array([10, 8, 5, 3, 1], dtype=np.int64)

    most_idx, most_s, rem_cps, least_s = get_slices_based_on_frequency(
        cps, most_sizes=(1, 1), frequency_array=freq_arr, least_sizes=(1,)
    )
    assert most_s == [[ord("A")], [ord("B")]]
    assert least_s == [[ord("E")]]
    assert rem_cps == [ord("C"), ord("D")]
    assert len(most_idx) == 2

    mat = np.ones((2, 5), dtype=np.float64)
    rem_cps2, rem_mat = preprocess_codepoint_list_and_lambda_matrix(
        most_sizes=(2,), frequency_array=freq_arr, codepoint_list=cps, lambda_matrix=mat
    )
    assert len(rem_cps2) == 3
    assert rem_mat.shape == (2, 3)

    strat = generate_most_and_equal_bin_strategy(
        cps, frequency_array=freq_arr, most_sizes=(2,), num_bins=2
    )
    assert strat.num_slices == 3
    assert strat.all_codepoints() == set(cps)


def test_online_lda_and_dnntm() -> None:
    cps = list(range(100, 112))
    docs = []
    for _ in range(30):
        docs.append("".join(chr(c) for c in cps[:6]))
        docs.append("".join(chr(c) for c in cps[6:]))
    proc = DocumentCorpusProcessor(docs)

    olda = OnlineLdaLearner(
        proc,
        cps,
        OnlineLdaConfig(n_topics=2, batch_size=16, n_batches=10, random_state=42),
    )
    olda_strat = olda.fit()
    assert olda_strat.num_slices == 2
    assert olda_strat.all_codepoints() == set(cps)
    assert olda.lambda_matrix_ is not None
    assert olda.lambda_matrix_.shape == (2, 12)

    for loss_type in (
        "REAL_FAKE_DIFF",
        "REAL_FAKE_AND_REAL_AVERAGE_DIFF",
        "ALL_COMBINED",
        "AUTOENCODER",
        "HYBRID_AUTOENCODER",
    ):
        dnntm = DnnTmLearner(
            proc,
            cps,
            DnnTmConfig(
                n_topics=2,
                n_layers=2,
                layer_size=16,
                batch_size=16,
                training_steps=15,
                loss_type=loss_type,
                random_seed=42,
            ),
        )
        dnn_strat = dnntm.fit()
        assert dnn_strat.num_slices == 2
        assert dnn_strat.all_codepoints() == set(cps)
        assert dnntm.lambda_matrix_ is not None
        assert dnntm.lambda_matrix_.shape == (2, 12)


def test_range_compact_ga_and_all_mutations() -> None:
    cps = list(range(1000, 1040))
    np_rng = np.random.default_rng(42)
    py_rng = random.Random(42)
    raw_weights = np_rng.uniform(0.1, 1.0, size=(4, 40))
    norm_weights = normalize_lambda_matrix(raw_weights, "topic_sum")

    g_mat = init_gene_from_topic_matrix(cps, norm_weights)
    g_bin = init_gene_by_equal_binning(cps, 4)
    assert g_bin.compute_costs(norm_weights)[0] == 4.0
    assert g_bin.check_no_topic_vanished()

    strat = g_bin.convert_to_strategy()
    g_re = init_gene_from_strategy(strat)
    assert g_re.compute_costs(norm_weights)[0] == 4.0

    for mut_fn in (
        purely_random_mutation,
        left_shift_mutation,
        right_shift_mutation,
        swap_topic_mutation,
        increase_min_topic_mutation,
        decrease_max_topic_mutation,
        merge_short_range_mutation,
    ):
        mutated = mut_fn(g_mat, rng=py_rng)
        assert mutated.n_codepoints == 40

    child = purely_random_crossover(g_mat, g_bin, rng=py_rng)
    assert child.n_codepoints == 40

    ga_cfg = RangeCompactConfig(
        range_cost_weight=0.02,
        max_iteration=60,
        n_survivals=15,
        n_mutations_per_iteration=30,
        n_crossovers_per_iteration=15,
        decay_max_diff=DecayConfig(
            min_val=4.0, max_val=10.0, decay_method="POWER", decay_power=2.0
        ),
        history_interval=20,
        random_seed=42,
    )
    learner = RangeCompactGeneticLearner(
        cps, raw_weights, ga_cfg, normalization_method="topic_sum"
    )
    ga_strat = learner.run()
    assert ga_strat.num_slices == 4
    assert ga_strat.all_codepoints() == set(cps)
    assert len(learner.history_) >= 3
    assert learner.history_[-1].best_final_cost <= learner.history_[0].best_final_cost + 1e-9

    from mlfont.range_compact import build_island_bridged_topics, potts_viterbi_smooth_topics

    t_potts = potts_viterbi_smooth_topics(cps, norm_weights, range_cost_weight=0.05)
    assert len(t_potts) == len(cps)
    assert set(t_potts.tolist()).issubset({0, 1, 2, 3})

    sparse_cps = list(range(0x4E00, 0x4E00 + 5200))
    sparse_mat = np_rng.uniform(0.0, 0.01, size=(10, 5200))
    rem_freq = np.zeros(5200, dtype=np.float64)
    for idx in range(0, 600, 2):
        rem_freq[idx] = 10.0
        sparse_mat[idx % 4, idx] = 1.0
    t_isl = build_island_bridged_topics(sparse_cps, sparse_mat, rem_frequency_array=rem_freq)
    assert t_isl is not None
    assert len(t_isl) == 5200
    assert np.all((t_isl >= 0) & (t_isl < 10))


def test_baselines_subsetter_and_evaluator(tmp_path: Path) -> None:
    cps = list(range(0x41, 0x5B))  # A..Z (26 codepoints)
    s_single = generate_single_slice_strategy(cps)
    s_min = generate_min_slice_strategy(cps)
    s_eq = generate_equal_bin_strategy(cps, num_slices=4)
    s_rand = generate_random_slice_strategy(cps, num_slices=4, seed=7)
    assert s_rand.all_codepoints() == set(cps)

    nam1 = tmp_path / "vowels.nam"
    save_codepoints_file([ord(c) for c in "AEIOU"], nam1)
    s_lang = generate_language_slice_strategy(cps, [nam1], include_unassigned=True)
    assert s_lang.num_slices == 2
    assert s_lang.all_codepoints() == set(cps)

    merged = merge_partial_strategies(
        [generate_single_slice_strategy(cps[:5], "part1")], full_codepoints=cps
    )
    assert merged.num_slices == 2
    assert merged.all_codepoints() == set(cps)

    uranges = format_unicode_ranges([0x41, 0x42, 0x43, 0x45, 0x47, 0x48])
    assert uranges == ["U+41-43", "U+45", "U+47-48"]

    ttf_path = _create_tiny_ttf(tmp_path / "TinyTest.ttf", cps)
    extracted_cps = extract_codepoints_from_font(ttf_path)
    assert extracted_cps == cps

    artifacts, css_path = slice_font_file(
        ttf_path, s_eq, tmp_path / "sliced", flavor="woff2"
    )
    assert len(artifacts) == 4
    assert all(a.byte_size > 0 and a.path.exists() for a in artifacts)
    assert "@font-face" in css_path.read_text(encoding="utf-8")

    samples = [
        {ord("A"), ord("B"), ord("C")},
        {ord("A"), ord("D"), ord("E")},
        {ord("F"), ord("G")},
        {ord("X"), ord("Y"), ord("Z")},
    ]
    evaluator = FontLoadingEvaluator(cps, samples)
    params = EvaluationParams(
        name="test_3g",
        rtt_ms=400.0,
        bandwidth_bytes_per_ms=50.0,
        average_glyph_size_bytes=200.0,
        session_length=2,
    )
    res_single = evaluator.evaluate_strategy(s_single, params)
    res_min = evaluator.evaluate_strategy(s_min, params)
    res_eq = evaluator.evaluate_strategy(s_eq, params)

    assert res_single.num_missing_codepoints == 0
    assert res_min.upfront_utilization_pct == 100.0
    assert res_min.session_utilization_pct == 100.0
    assert res_eq.num_ranges == 4

    out_eval_json = tmp_path / "eval_results.json"
    save_evaluation_results_json([res_single, res_min, res_eq], out_eval_json)
    assert out_eval_json.exists()

    for preset_name in ("chinese", "traditional_chinese", "japanese", "korean", "lgc"):
        preset = get_preset_config(preset_name)
        assert preset.olda_config.n_topics > 0

    bundle_dir = export_pretrained_bundle(
        output_dir=tmp_path / "pretrained_bundle",
        strategy=s_eq,
        lambda_matrix=np.ones((4, 26), dtype=np.float64),
        codepoints=cps,
        frequency_array=np.arange(26, dtype=np.int64),
        font_path=ttf_path,
        font_family="TinyTest Sliced",
        slice_font=True,
        evaluation_results=[res_eq],
    )
    loaded_bundle = load_pretrained(bundle_dir)
    assert loaded_bundle.strategy.num_slices == 4
    assert loaded_bundle.lambda_matrix is not None
    assert loaded_bundle.lambda_matrix.shape == (4, 26)
    assert loaded_bundle.predict_required_slices("AZ") == [0, 3]


def test_sparse_vocabulary_max_diff_and_aliases(tmp_path: Path) -> None:
    import mlfont

    for sym in (
        "DNNTMConfig",
        "DNNTMSolver",
        "FontSubsetter",
        "FrequencySolver",
        "OnlineLDAConfig",
        "OnlineLDASolver",
        "RangeCompactGASolver",
        "extract_font_codepoints",
        "normalize_topic_matrix",
        "save_safetensors_model",
    ):
        assert hasattr(mlfont, sym), f"Missing public symbol {sym}"

    # 120 codepoints where only 40 are observed in training documents
    cps = list(range(0x4E00, 0x4E00 + 120))
    docs = []
    for i in range(40):
        start_c = (i % 4) * 10
        end_c = ((i % 4) + 1) * 10
        cluster = cps[start_c:end_c]
        docs.append("".join(chr(c) for c in cluster))
    proc = DocumentCorpusProcessor(docs, random_seed=42)

    olda = OnlineLdaLearner(
        proc,
        cps,
        OnlineLdaConfig(
            n_topics=6,
            most_sizes=[12],
            normalization_method="global_max",
            batch_size=20,
            n_batches=5,
            random_state=42,
        ),
    )
    s_olda = olda.fit()
    assert s_olda.num_slices == 7  # 1 most_slice + 6 topic slices
    assert olda.lambda_matrix_ is not None
    assert olda.lambda_matrix_.shape == (6, 108)
    # All 6 topic rows carry real mass (> 1e-3) and non-constant learned signal on active cols
    assert int(np.sum(np.any(olda.lambda_matrix_ > 1e-3, axis=1))) == 6
    assert int(np.sum(np.ptp(olda.lambda_matrix_[:, :28], axis=1) > 1e-4)) == 6
    assert int(np.sum(np.max(olda.lambda_matrix_, axis=0) <= 1e-3)) == 0

    dnntm = DnnTmLearner(
        proc,
        cps,
        DnnTmConfig(
            n_topics=6,
            most_sizes=[12],
            normalization_method="global_max",
            batch_size=16,
            n_layers=2,
            layer_size=16,
            training_steps=20,
            loss_type="ALL_COMBINED",
            random_seed=42,
        ),
    )
    s_dnn = dnntm.fit()
    assert s_dnn.num_slices == 7
    assert all(loss_val >= 0.0 for loss_val in dnntm.loss_history_)

    freq = proc.count_frequency_array(cps)
    ga_cfg = RangeCompactConfig(
        range_cost_weight=0.005,
        max_iteration=80,
        n_survivals=15,
        n_mutations_per_iteration=30,
        n_crossovers_per_iteration=15,
        decay_max_diff=DecayConfig(
            min_val=3.0, max_val=15.0, decay_method="POWER", decay_power=3.0
        ),
        random_seed=42,
    )
    ga = RangeCompactGeneticLearner(
        cps,
        olda.lambda_matrix_,
        ga_cfg,
        frequency_array=freq,
        most_sizes=[12],
        normalization_method="global_max",
    )
    s_ga = ga.run()
    topic_slice_sizes = [len(s) for s in s_ga.subsets[1:]]
    assert max(topic_slice_sizes) - min(topic_slice_sizes) <= 3


def test_readme_quickstart_api_and_edge_cases(tmp_path: Path) -> None:
    from mlfont import (
        DNNTMConfig,
        DNNTMSolver,
        FontSubsetter,
        FrequencySolver,
        OnlineLDAConfig,
        OnlineLDASolver,
        RangeCompactConfig,
        RangeCompactGASolver,
        extract_font_codepoints,
    )

    cps = list(range(0x0041, 0x0041 + 60))
    font_path = _create_tiny_ttf(tmp_path / "Quickstart.ttf", cps)
    font_codepoints = extract_font_codepoints(font_path)
    assert len(font_codepoints) == 60

    train_docs = [
        set(font_codepoints[:20]),
        set(font_codepoints[15:35]),
        set(font_codepoints[30:50]),
    ] * 10

    freq_solver = FrequencySolver(font_codepoints)
    freq_solver.fit(train_docs)
    most_slices, remaining_codepoints = freq_solver.partition_most_frequent(
        most_counts=[12]
    )
    assert len(most_slices) == 1
    assert len(most_slices[0]) == 12
    assert len(remaining_codepoints) == 48

    olda_solver = OnlineLDASolver(
        codepoints=remaining_codepoints,
        config=OnlineLDAConfig(
            n_topics=4,
            batch_size=16,
            max_iter=5,
            normalization="global_max",
            random_state=42,
        ),
    )
    topic_matrix = olda_solver.fit_transform(train_docs)
    assert topic_matrix.shape == (4, 48)

    dnntm_solver = DNNTMSolver(
        codepoints=remaining_codepoints,
        config=DNNTMConfig(
            n_topics=4,
            batch_size=16,
            max_iter=10,
            normalization="global_max",
            random_state=42,
        ),
    )
    dnn_matrix = dnntm_solver.fit_transform(train_docs)
    assert dnn_matrix.shape == (4, 48)

    # Test exact max_diff=0 when 48 % 4 == 0
    ga_solver = RangeCompactGASolver(
        codepoints=remaining_codepoints,
        topic_matrix=topic_matrix,
        config=RangeCompactConfig(
            population_size=10,
            num_iterations=40,
            decay_power=3.0,
            range_cost_weight=0.012,
            decay_max_diff=DecayConfig(min_val=0.0, max_val=4.0),
            random_state=42,
        ),
    )
    strategy = ga_solver.optimize(
        strategy_name="range_compact_most12_topics4",
        font_name="Quickstart",
        prefix_slices=most_slices,
    )
    assert strategy.num_slices == 5
    assert strategy.total_num_ranges() == strategy.num_ranges
    ga_sizes = [len(s) for s in strategy.subsets[1:]]
    assert max(ga_sizes) - min(ga_sizes) == 0

    subsetter = FontSubsetter(font_path)
    artifacts, css_path = subsetter.slice_font(strategy, tmp_path / "out_woff2")
    assert len(artifacts) == 5
    assert css_path.exists()

    # Edge case: V_active < K (only 2 active codepoints, K=6 topics)
    tiny_docs = [
        "".join(chr(c) for c in remaining_codepoints[:2]),
        "".join(chr(c) for c in remaining_codepoints[:2]),
    ]
    sparse_solver = OnlineLDASolver(
        codepoints=remaining_codepoints,
        config=OnlineLDAConfig(n_topics=6, batch_size=4, max_iter=3, random_state=42),
    )
    sparse_strat = sparse_solver.fit(tiny_docs)
    assert sparse_strat.num_slices == 6
    assert sparse_strat.all_codepoints() == set(remaining_codepoints)


def test_cli_scripts_end_to_end(tmp_path: Path, monkeypatch) -> None:
    import sys
    from scripts import evaluate, export_model, inference, prepare_data, train

    font_path = _create_tiny_ttf(tmp_path / "CliFont.ttf", list(range(0x0041, 0x0041 + 40)))
    data_dir = tmp_path / "data"
    train_dir = tmp_path / "trained"
    export_dir = tmp_path / "exported"
    eval_json = tmp_path / "eval_results.json"
    slice_dir = tmp_path / "sliced"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-prepare-data",
            "--font_path",
            str(font_path),
            "--generate_synthetic",
            "--num_synthetic",
            "40",
            "--output_dir",
            str(data_dir),
        ],
    )
    prepare_data.main()
    assert (data_dir / "codepoints.nam").exists()
    assert (data_dir / "train.jsonl").exists()
    assert (data_dir / "test.jsonl").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-train",
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(train_dir),
            "--n_topics",
            "4",
            "--most_sizes",
            "8",
            "--max_iteration",
            "15",
        ],
    )
    train.main()
    assert (train_dir / "strategy.json").exists()
    assert (train_dir / "model.safetensors").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-evaluate",
            "--codepoints_file",
            str(data_dir / "codepoints.nam"),
            "--test_jsonl",
            str(data_dir / "test.jsonl"),
            "--strategy_files",
            str(train_dir / "strategy.json"),
            "--output_json",
            str(eval_json),
        ],
    )
    evaluate.main()
    assert eval_json.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-export",
            "--checkpoint_dir",
            str(train_dir),
            "--export_dir",
            str(export_dir),
            "--font_path",
            str(font_path),
        ],
    )
    export_model.main()
    assert (export_dir / "model.safetensors").exists()
    assert (export_dir / "strategy.json").exists()
    assert (export_dir / "font_sliced.css").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-slice",
            "--model_dir",
            str(export_dir),
            "--font_path",
            str(font_path),
            "--output_dir",
            str(slice_dir),
            "--text",
            "ABCDEF",
        ],
    )
    inference.main()
    assert (slice_dir / "clifont_sliced.css").exists()
