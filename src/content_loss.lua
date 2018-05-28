local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize, mask)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.mask = mask
  -- print(mask)
end

function ContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    -- print(input:size(), self.mask:size())
    self.loss = self.crit:forward(torch.cmul(input, self.mask), torch.cmul(self.target, self.mask)) * self.strength
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = torch.cmul(self.crit:backward(input, self.target), self.mask)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end
