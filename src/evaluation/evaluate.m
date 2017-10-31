function result = evaluate(output_dir, target_dir, task)
  ts = get_task(output_dir, target_dir, task);
  [p, s, a] = run_eval_template(ts{k,2},ts{k,3},ts{k,4},get_path(ts{k,5}),ts{k,6},ts{k,7});
  result = [ts{k,1} ' psnr,ssim,abs= ' num2str(p) ', ' num2str(s) ', ' num2str(a)];
  end
  result
end