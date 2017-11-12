function result = evaluate(output_dir, target_dir, task)
  ts = get_task(output_dir, target_dir, task);
  [p, s, a] = run_eval_template(ts{1,2},ts{1,3},ts{1,4},get_path(ts{1,5}),ts{1,6},ts{1,7});
  result = [ts{1,1} ' psnr,ssim,abs= ' num2str(p) ', ' num2str(s) ', ' num2str(a)];
  result
end
