
class Learner:
    def __init__(self, conf):
        self.lambda_update_freq = conf.lambda_update_freq
        self.bic_loss_to_start_change = conf.bic_loss_to_start_change
        self.lambda_bicubic_decay_rate = conf.lambda_bicubic_decay_rate
        self.bic_loss_counter = 0
        self.lambda_bicubic_min = 5e-6
        self.update_l_rate_freq = conf.lr_update_freq
        self.update_l_rate_rate = conf.lr_update_rate
        self.lambda_sparse_end = conf.lambda_sparse_end
        self.lambda_centralized_end = conf.lambda_centralized_end
        self.lambda_negative_end = conf.lambda_negative_end
        self.similar_to_bicubic = False   # Determines if similarity to bicubic downscaling is satisfied
        self.insert_constraints = True    # Switch to false after constraints are inserted

    def update(self, iteration, gan, logger):
        if iteration == 0:
            return
        # Update learning rate every update_l_rate freq
        if iteration % self.update_l_rate_freq == 0:
            for params in gan.optimizer_G.param_groups:
                params['lr'] /= self.update_l_rate_rate
                logger.important_log([self.update_l_rate_rate, params['lr'], iteration], category='LR_G')
            for params in gan.optimizer_D.param_groups:
                params['lr'] /= self.update_l_rate_rate
                logger.important_log([self.update_l_rate_rate, params['lr'], iteration], category='LR_D')

        # Until similar to bicubic is satisfied, don't update any other lambdas
        if not self.similar_to_bicubic:
            if gan.loss_bicubic < self.bic_loss_to_start_change:
                if self.bic_loss_counter >= 2:
                    self.similar_to_bicubic = True
                    logger.important_log('Started to update Lambda\'s on iteration %d at Bic loss %.2f' % (iteration, gan.loss_bicubic))
                else:
                    self.bic_loss_counter += 1
            else:
                self.bic_loss_counter = 0
        # Once similar to bicubic is satisfied, consider inserting other constraints
        elif iteration % self.lambda_update_freq == 0 and gan.lambda_bicubic > self.lambda_bicubic_min:
            gan.lambda_bicubic = max(gan.lambda_bicubic/self.lambda_bicubic_decay_rate, self.lambda_bicubic_min)
            if self.insert_constraints and gan.lambda_bicubic < 5e-3:
                gan.lambda_centralized = self.lambda_centralized_end
                gan.lambda_sparse = self.lambda_sparse_end
                gan.lambda_negative = self.lambda_negative_end
                logger.important_log('Lambda Centralized update: 0 --> %.1f on iteration %d' % (gan.lambda_centralized, iteration))
                logger.important_log('Lambda Sparse update: 0 --> %.1f on iteration %d' % (gan.lambda_sparse, iteration))
                self.insert_constraints = False
            logger.important_log('Lambda Bicubic update: %.0e --> %.0e on iteration %d' % (gan.lambda_bicubic * 100, gan.lambda_bicubic, iteration))


class LRPolicy(object):
    def __init__(self, start, end):
        self.start = start
        self.duration = end - start

    def __call__(self, cur_iter):
        return 1. - max(0., float(cur_iter-self.start)) / float(self.duration)
