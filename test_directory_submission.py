import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

import app
import engine as e


class DirectorySubmissionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        self.src, self.out = base / "in", base / "out"
        self.out.mkdir()
        for name in (
            "集合/a.mp4",
            "集合/子目录/a.MKV",
            "集合/说明.txt",
            "集合/@eaDir/x.mp4",
            "其他/b.mov",
        ):
            file = self.src / name
            file.parent.mkdir(exist_ok=True, parents=True)
            file.touch()
        self.enterContext(
            patch.multiple(
                e,
                INPUT=self.src,
                OUTPUT=self.out,
                STATE=base,
                INPUT_DISPLAY=str(self.src),
                OUTPUT_DISPLAY=str(self.out),
            )
        )
        e.initialize()

    def test_live_progress_switches_from_diffusion_to_stereo_writes(self):
        (e.STATE / "logs").mkdir()
        key = e.enqueue(self.src / "集合/a.mp4", mode="VR180 110°", auto_stage=True)
        with e.db() as c:
            c.execute("UPDATE jobs SET status='running' WHERE id=?", (key,))
        log = e.set_stage(key, "VR360 A 连续原帧扩画", 3, 100)
        with log.open("a", encoding="utf8") as f:
            f.write("VR360 A shot-0-0 frame=100\n")
        self.assertIn("100/100", app.live_progress()[0])
        folder = e.STATE / f"work-{key}-test"
        folder.mkdir()
        (folder / "stage.mkv").write_bytes(b"frame")
        html = app.live_progress()[0]
        self.assertIn("正在进行双眼边缘合成", html)
        self.assertNotIn("100/100", html)
        self.assertIsNone(app.stereo_output_progress(key, (folder / "stage.mkv").stat().st_mtime + 1, True))

    def test_both_presets_use_gpu_and_allow_manual_cpu_selection(self):
        opts = {"mode": "VR180 110°", "auto_stage": True}
        src = self.src / "集合/a.mp4"
        normal, saved = e.prepare_job(src, **opts)
        quality, quality_spec = e.prepare_job(src, **opts, **app.SPEED_PRESETS["质量"])
        self.assertTrue(json.loads(quality_spec)["vr360"]["detector_cuda"])
        self.assertEqual((normal, saved), (quality, quality_spec))
        key = e.enqueue(src, **opts, **app.SPEED_PRESETS["快速"])
        self.assertEqual(key, quality)
        manual = {**app.SPEED_PRESETS["质量"], "detector_cuda": False}
        cpu_key = e.enqueue(src, **opts, **manual)
        self.assertNotEqual(key, cpu_key)
        self.assertFalse(json.loads(app.get_job(cpu_key)["spec"])["vr360"]["detector_cuda"])
        with e.db() as connection:
            settings = json.loads(connection.execute("select spec from jobs where id=?", (key,)).fetchone()[0])["vr360"]
        for name, value in app.SPEED_PRESETS["快速"].items():
            self.assertEqual(settings[name], value)
        self.assertTrue(settings["protect_actors"])
        self.assertTrue(settings["contour_enabled"])

    def test_editable_fields_show_their_default_and_help(self):
        components = app.build().get_config_file()["components"]
        fields = [c["props"] for c in components
                  if c["type"] in {"checkbox", "dropdown", "number", "radio", "slider", "textbox"}
                  and c["props"].get("label") and c["props"].get("interactive") is not False]
        labels = {p["label"] for p in fields}
        self.assertTrue({entry[0] for entry in app.CONTROLS.values()} <= labels)
        for field in fields:
            with self.subTest(field=field["label"]):
                if field["label"] in {"输入根目录", "输出根目录", "本次处理目录（含子目录）", "单个视频（可选）"}:
                    self.assertTrue(field.get("info"))
                    self.assertNotIn("默认值", field["info"])
                    continue
                prefix, _, description = field.get("info", "").partition("。")
                self.assertTrue(prefix.startswith("默认值为 "))
                self.assertTrue(description.strip())

    def submit(self, target, **options):
        return app.submit(
            target,
            3.5,
            0,
            "VDA_S",
            mode="VR180 100°",
            auto_stage=True,
            stage_preset=e.CLOSEUP,
            output_path=self.out,
            **options,
        )

    def test_recursive_batch_is_scoped_and_retry_reuses_single_file_job(self):
        _, first = self.submit(
            self.src, listing_root="集合", single_video=str(self.src / "集合/a.mp4")
        )
        summary, key = self.submit(self.src, listing_root="集合")
        self.assertEqual(first, key)
        self.assertIn("本次入队 1 个", summary)
        specs = [json.loads(row["spec"]) for row in e.rows()]
        jobs = {s["rel"]: s for s in specs}
        self.assertEqual(set(jobs), {"集合/a.mp4", "集合/子目录/a.MKV"})
        nested = jobs["集合/子目录/a.MKV"]
        self.assertEqual(e.job_output_folder(nested), self.out / "集合/子目录")
        self.assertEqual(nested["stage_version"], e.VR360_A)
        self.assertTrue(all(spec["vr360"]["source_mask_enabled"] for spec in specs))
        _, disabled = self.submit(self.src / "集合/a.mp4", source_mask_enabled=False)
        self.assertFalse(json.loads(app.get_job(disabled)["spec"])["vr360"]["source_mask_enabled"])
        self.assertIn("本次入队 0 个", self.submit(self.src, listing_root="集合")[0])

    def test_custom_input_root_and_nested_scope(self):
        root = self.src / "集合"
        summary, key = self.submit(root, listing_root="子目录")
        self.assertIn(str(self.out / "子目录"), summary)
        self.assertEqual(
            e.job_output_folder(json.loads(app.get_job(key)["spec"])),
            self.out / "子目录",
        )
        with self.assertRaisesRegex(gr.Error, "本次处理目录"):
            self.submit(root, listing_root="子目录", single_video=str(root / "a.mp4"))
        with self.assertRaisesRegex(gr.Error, "输入根目录"):
            self.submit(root, listing_root="../其他")
        self.assertEqual(len(e.rows()), 1)

    def test_batch_skips_a_full_existing_output_for_the_same_projection(self):
        output = self.out / "其他"
        output.mkdir()
        existing = output / "b_C_MetricL_12345678_Center_Curve50_VR360A_FOV100_LR_180.mp4"
        existing.write_bytes(b"verified output")
        picture = {"codec_type": "video", "codec_name": "hevc", "width": 5760, "height": 2880,
                   "duration": "42.0", "side_data_list": [
                       {"side_data_type": "Stereo 3D"}, {"side_data_type": "Spherical Mapping"}]}
        audio = {"codec_type": "audio"}
        metadata = {"format": {"duration": "42"}, "streams": [picture, audio]}
        with patch.object(e, "probe", return_value=metadata):
            message, key = self.submit(self.src, listing_root="其他")
            self.assertIn("已有成片跳过 1 个", message)
            self.assertIsNone(key)
            self.assertEqual(e.rows(), [])
            _, single = self.submit(self.src / "其他/b.mov")
            self.assertEqual(e.rows()[0]["id"], single)

    def test_batch_does_not_skip_a_short_output_or_a_different_fov(self):
        output = self.out / "其他"
        output.mkdir()
        short = output / "b_A_a1234567_FOV100_LR_180.mp4"
        wrong_fov = output / "b_A_a1234568_FOV110_LR_180.mp4"
        short.write_bytes(b"short sample")
        wrong_fov.write_bytes(b"another view")
        video = {"codec_type": "video", "codec_name": "hevc", "width": 5760, "height": 2880,
                 "duration": "10", "side_data_list": [
                     {"side_data_type": "Stereo 3D"}, {"side_data_type": "Spherical Mapping"}]}
        source_meta = {"format": {"duration": "42"}, "streams": []}
        with patch.object(e, "probe", side_effect=lambda path: {
            "format": {"duration": "10"}, "streams": [video]
        } if path == short else source_meta) as probe:
            message, key = self.submit(self.src, listing_root="其他")
            self.assertNotIn(wrong_fov, [call.args[0] for call in probe.call_args_list])
        self.assertIn("本次入队 1 个", message)
        self.assertIsNotNone(key)

    def test_unified_vr360_settings(self):
        options = dict(mode="VR180 100°", auto_stage=True)
        plain = e.prepare_job("其他/b.mov", **options)
        for preset in (e.CLOSEUP, e.DANCE):
            self.assertEqual(plain, e.prepare_job("其他/b.mov", stage_preset=preset, **options))
        for change in (dict(seed=0), dict(match_stage=False), dict(seam_luma=False),
                       dict(observed_edges=False), dict(anime_detail=False), dict(contour_enabled=False), dict(contour_strength=0.5), dict(contour_width=32), dict(texture_enabled=False), dict(union_enabled=False), dict(depth_attention=.6), dict(palette_enabled=True), dict(generation_fps=12)):
            self.assertNotEqual(plain, e.prepare_job("其他/b.mov", **options, **change))
        self.assertEqual(json.loads(plain[1])["stage_version"], e.VR360_A)
        with self.assertRaises(ValueError):
            e.prepare_job("其他/b.mov", **options, seed=0.5)

    def test_root_and_flat_output(self):
        self.submit(".", keep_folders=False)
        self.assertEqual(len(e.rows()), 3)
        for row in e.rows():
            self.assertEqual(e.job_output_folder(json.loads(row["spec"])), self.out)

    def test_extended_controls_roundtrip_in_single_and_directory_submissions(self):
        import inspect
        from vr360_options import BASE_FIELDS, CONTROLS, GROUPS

        visible = [key for group in GROUPS.values() for key in group]
        self.assertCountEqual(visible, [*BASE_FIELDS, *CONTROLS])
        params = list(inspect.signature(app.submit).parameters)
        self.assertEqual(params[params.index("union_enabled"):],
                         [*(k for k in CONTROLS if k not in ("front_only", "union_variant")), "outpaint_model",
                          *app.H3_FIELDS, "front_only", "view_zoom", "union_variant"])
        changes = dict(guide_extension_enabled=False, guide_extension_width=48, guide_extension_margin=20,
                       guide_extension_blur=4, guide_extension_contrast=50,
                       seam_luma=False, seam_strength=.65, observed_edges=False, observed_strength=.3,
                       anime_detail=False, detail_strength=.25, source_mask_enabled=True, source_mask_inside=.85, source_mask_outside=.15,
                       source_attention=.75, union_enabled=False, steps=6, generation_fps=16,
                       detector_cuda=True, iw3_batch=2, offload_streams=4, joint_audio=False, png_compression=0,
                       front_only=False)
        for item in ("集合/a.mp4", "其他"):
            _, key = self.submit(self.src / item, **changes)
            saved = json.loads(app.get_job(key)["spec"])["vr360"]
            self.assertEqual({k: saved[k] for k in changes}, changes)

    def test_projection_and_front_only_save_for_a_new_job(self):
        _, key = self.submit(self.src / "集合/a.mp4", curvature=100)
        spec = json.loads(app.get_job(key)["spec"])
        self.assertEqual((spec["curvature"], spec.get("view_zoom", 1.0), spec["vr360"]["front_only"]), (100, 1.0, True))
        _, zoomed = self.submit(self.src / "集合/a.mp4", curvature=100, view_zoom=1.10)
        self.assertEqual(json.loads(app.get_job(zoomed)["spec"])["view_zoom"], 1.10)
        for zoom in (0.9, 1.26, float("nan"), True):
            with self.assertRaisesRegex(ValueError, "正前方放大"):
                e.prepare_job("其他/b.mov", mode="VR180 110°", view_zoom=zoom)

    def test_ltx25_uses_the_shared_ltx_settings_when_installed(self):
        with patch("vr360_options.ltx25_available", return_value=True):
            _, key = self.submit(self.src / "集合/a.mp4", outpaint_model="ltx25",
                                 steps=12, source_mask_outside=0.2, h3_union=False)
        opts = json.loads(app.get_job(key)["spec"])["vr360"]
        self.assertEqual((opts["outpaint_model"], opts["steps"], opts["source_mask_outside"]),
                         ("ltx25", 12, 0.2))
        self.assertNotIn("h3_union", opts)

    def test_ltx_quality_preset_persists_syfe_and_fast_reuses_official_defaults(self):
        with patch("vr360_options.ltx25_available", return_value=True), \
                patch("vr360_options.syfe_union_path", return_value=Path("syfe.safetensors")):
            for model in ("ltx2", "ltx25"):
                fast_model, fast = app.generation_profile("快速", model)
                quality_model, quality = app.generation_profile("质量", model)
                self.assertEqual((fast_model, quality_model), (model, model))
                self.assertEqual(fast, dict(quality, union_variant="official"))
                a, default_spec = e.prepare_job("其他/b.mov", mode=fast["mode"], auto_stage=True,
                                                outpaint_model=model)
                b, fast_spec = e.prepare_job("其他/b.mov", mode=fast["mode"], auto_stage=True,
                                             outpaint_model=model, union_variant=fast["union_variant"])
                self.assertEqual((a, default_spec), (b, fast_spec))
                _, key = self.submit(self.src / "集合/a.mp4", outpaint_model=model,
                                     union_variant=quality["union_variant"])
                opts = json.loads(app.get_job(key)["spec"])["vr360"]
                self.assertEqual(opts["union_variant"], "syfe")
        with patch("vr360_options.syfe_union_path", return_value=None):
            with self.assertRaisesRegex(ValueError, "缺少 SyFe"):
                e.prepare_job("其他/b.mov", mode="VR180 110°", auto_stage=True, union_variant="syfe")

    def test_h3_routing_and_inactive_parameters(self):
        for enabled in (False, True):
            _, key = self.submit(self.src / "集合", outpaint_model="h3", h3_union=enabled,
                                 h3_size=512, h3_steps=20, union_enabled=True, steps=2)
            spec = json.loads(app.get_job(key)["spec"])
            opts = spec["vr360"]
            self.assertEqual(opts["h3_union"], enabled)
            self.assertEqual(opts["h3_steps"], 20)
            self.assertFalse(app.LTX_ONLY & opts.keys())
            import vr360_h3
            graph = vr360_h3.graph(opts, "shot-0-0")
            self.assertEqual("union" in graph["13"]["inputs"], enabled)
            self.assertEqual("15" in graph, enabled)
            self.assertEqual(len([n for n in graph.values() if n["class_type"] == "H3SphereOutpaint"]), 1)
        _, hq = self.submit(self.src / "集合", outpaint_model="h3_hq", h3_union=True)
        opts = json.loads(app.get_job(hq)["spec"])["vr360"]
        self.assertEqual(opts["outpaint_model"], "h3")
        self.assertEqual(vr360_h3.graph(opts, "shot-0-0")["8"]["inputs"]["unet_name"],
                         "MiniMax-H3_FL2VA-NVFP4-HQ.safetensors")
        self.assertNotEqual(key, hq)

    def test_custom_model_choice_and_job_use_manifest_capabilities(self):
        from vr360_options import CUSTOM_ASSET_DIRS
        import stage_vr360
        import vr360_h3

        root = e.STATE / "models"
        self.enterContext(patch.dict("os.environ", {"NUNIF_HOME": str(root)}))
        for family in ("ltx2", "ltx25", "h3"):
            manifest_path = root / "outpaint" / family / "model.json"
            roles = ["checkpoint", "video_vae", "text_encoder"]
            if family != "h3":
                roles.append("outpaint_lora")
            if family == "ltx2":
                roles.append("text_projection")
            assets = {role: f"test-{family}-{role}.safetensors" for role in roles}
            for role, name in assets.items():
                file = manifest_path.parent / CUSTOM_ASSET_DIRS[role] / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"fixture")
            steps = "h3_steps" if family == "h3" else "steps"
            manifest = {"label": f"Example {family}", "family": family, "assets": assets,
                        "features": [], "settings": {steps: 12}}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            selection = f"custom:{family}"
            self.assertIn((f"Example {family} · 自定义", selection), app.available_model_choices())
            _, key = self.submit(
                self.src / "集合/a.mp4", outpaint_model=selection, union_enabled=True, union_variant="syfe",
                h3_union=True, source_mask_enabled=True, source_attention=.5, guide_extension_enabled=True,
                palette_enabled=True, vae_tiled=True, detector_cuda=True, joint_audio=True,
            )
            spec = json.loads(app.get_job(key)["spec"])
            opts = spec["vr360"]
            self.assertIn(f"Example {family}", app.mode_label(spec))
            self.assertEqual(opts[steps], 12)
            self.assertEqual(opts["custom_model"]["assets"], assets)
            for field in ("source_mask_enabled", "guide_extension_enabled", "palette_enabled", "detector_cuda"):
                self.assertFalse(opts[field])
            if family == "h3":
                graph = vr360_h3.graph(opts, "window")
                self.assertNotIn("15", graph)
                self.assertNotIn("union", graph["13"]["inputs"])
                self.assertEqual(graph["8"]["inputs"]["unet_name"], assets["checkpoint"])
                self.assertEqual(graph["6"]["inputs"]["vae_name"], assets["video_vae"])
                self.assertEqual(graph["7"]["inputs"]["clip_name"], assets["text_encoder"])
            else:
                graph = json.loads((e.ROOT / "vr360_graph.json").read_bytes())
                stage_vr360.configure_custom_ltx(graph, opts["custom_model"], opts["prompt"])
                stage_vr360.apply_inference_controls(graph, "window", opts)
                for field in ("union_enabled", "joint_audio", "vae_tiled", "front_only"):
                    self.assertFalse(opts[field])
                self.assertEqual(graph["1"]["inputs"]["unet_name"], assets["checkpoint"])
                self.assertEqual(graph["2"]["inputs"]["lora_name"], assets["outpaint_lora"])
                self.assertNotIn("4", graph)
                self.assertNotIn("source_weights", graph)
                self.assertEqual(graph["12"]["inputs"]["width"], 1280)

            manifest["settings"][steps] = 16
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            _, changed = self.submit(self.src / "集合/a.mp4", outpaint_model=selection)
            self.assertNotEqual(key, changed)
            self.assertEqual(json.loads(app.get_job(key)["spec"])["vr360"][steps], 12)
            manifest["assets"]["checkpoint"] = "../hidden.safetensors"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(gr.Error, "模型文件名"):
                self.submit(self.src / "集合/a.mp4", outpaint_model=selection)

    def test_empty_or_invalid_batch_does_not_queue_partial_work(self):
        (self.src / "空").mkdir()
        with self.assertRaisesRegex(gr.Error, "没有支持的视频"):
            self.submit("空")
        with self.assertRaises(ValueError):
            e.enqueue_many(["集合/a.mp4", "消失.mp4"])
        self.assertEqual(e.rows(), [])
